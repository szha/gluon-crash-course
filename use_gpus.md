# Use GPUs

We often use GPU to train and deploy neural networks, because it often offers significant more computation power compared to a CPU. This tutorial we will introduce how to use GPUs in MXNet. 

First, make sure you have at least one Nvidia GPU in your machine and CUDA properly installed. Other GPUs such as AMD and Intel GPUs are not supported yet. Then check you installed the GPU-enabled MXNet.

```{.python .input  n=15}
# If you pip installed the plain `mxnet` before, uncomment the 
# following two lines to install the GPU version. You may need to 
# replace `cu91` according to your CUDA version. 
#
# !pip uninstall mxnet
# !pip install mxnet-cu91

from mxnet import nd, gpu, gluon, autograd
from mxnet.gluon import nn
from time import time
```

## Allocate data on a GPU

You may notice that MXNet's NDArray is very similar to Numpy. One major difference is NDArray has a `context` attribute that specifies which device this array is on. In default it is `cpu()`. Now we change it to the first GPU.

```{.python .input  n=10}
x = nd.ones((3,4), ctx=gpu())
x
```

For CPU, MXNet will allocate data on main memory and try to use all CPU cores as possible even if there is more than one CPU socket. While if there are multiple GPUs, MXNet needs to specify which GPUs the NDArray will be allocated. 

Assume there is another GPU, we can create another NDArray here. (If you only have one GPU, then you will see an error). Here we copy `x` to GPU 1:

```{.python .input  n=11}
x.copyto(gpu(1))
```

MXNet needs users to explicitly move data between devices. But several operators such as `print`, `asnumpy` and `asscalar`, will implicit move data to main memory. 

## Run operation on a GPU

To perform an operation on a particular GPU, we only need to guarantee that the inputs of this operation are already on that GPU. The output will be allocated on the same GPU as well. Almost all operators in the `nd` module support run on GPU.

```{.python .input  n=21}
y = nd.random.uniform(shape=(3,4), ctx=gpu())
x + y
```

Remember that if the inputs are not on the same GPU, you will see an error. 

## Run a neural network on a GPU

Simliar, to run a nueral network on a GPU, we only need to copy/move the input data and parameters to the GPU. Let reuse the previous defined LeNet

```{.python .input  n=16}
net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(120, activation="relu"),
        nn.Dense(84, activation="relu"),
        nn.Dense(10)
    )
```

And then load the saved parameters into GPU 0 directly, or use `net.collect_params().reset_ctx` to change the device.

```{.python .input  n=20}
net.load_params('net.params', ctx=gpu(0))
```

Now create an input data on GPU 0, the forward function will then run on GPU 0.

```{.python .input  n=22}
x = nd.random.uniform(shape=(1,1,28,28), ctx=gpu(0))
net(x)
```

## Multi-GPU training (advance)

Finally, we show how to use multiple GPUs to jointly train a neural network through data parallelism. Assume there is *n* GPUs, we then split each data batch into *n* parts, and each GPU will run the forward and backward using one part data. 

Let's first copy the data definition and transform from the previous tutorial.

```{.python .input}
batch_size = 256
mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_valid = gluon.data.vision.FashionMNIST(train=False)
valid_data = gluon.data.DataLoader(
    mnist_valid, batch_size, shuffle=False)
train_data = gluon.data.DataLoader(
    mnist_train, batch_size, shuffle=True)

def transform(data, label):
    return data.transpose((0,3,1,2)).astype('float32')/255, label.astype('float32')
```

The training loop is quite similar to that we introduced before. Major differences are highlighted in the codes.

```{.python .input}
# Diff 1: Use two GPUs for training.
devices = [gpu(0), gpu(1)]

# Diff 2: reinitialize the parameters and place them on multiple GPUs
net.collect_params().initialize(force_reinit=True, ctx=devices)

# Loss and trainer as before
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.1})

for epoch in range(10):
    train_loss = 0.
    tic = time()
    for data, label in train_data:
        data, label = transform(data, label)
        
        # Diff 3: split batch and load into corresponding devices
        data_list = gluon.utils.split_and_load(data, devices)
        label_list = gluon.utils.split_and_load(label, devices)

        # Diff 4: run forward and backward on each devices. 
        # MXNet will automatically run them in parallel
        with autograd.record():
            losses = [softmax_cross_entropy(net(X), y)
                      for X, y in zip(data_list, label_list)]
        for l in losses:
            l.backward()
            
        trainer.step(batch_size)
        
        # Diff 5: sum losses over all devices 
        train_loss += sum([l.sum().asscalar() for l in losses])

    print("Epoch %d: Loss: %.3f, Time %.1f sec" % (
        epoch, train_loss/len(train_data)/batch_size, time()-tic))
```
