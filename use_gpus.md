# Use GPUs

We often use GPU to train and deploy neural networks, because it often offers significant more computation power compared to a CPU. This tutorial we will introduce how to use GPUs in MXNet. 

First, make sure you have at least one Nvidia GPU in your machine and CUDA properly installed. Other GPUs such as AMD and Intel GPUs are not supported yet. Then check you installed the GPU-enabled MXNet. 

```{.python .input  n=2}
# If you pip installed the plain `mxnet` before, uncomment the following 
# two lines to install the GPU version. You may need to replace `cu91` 
# according to your CUDA version. 
#
# !pip uninstall mxnet
# !pip install mxnet-cu91

from mxnet import nd, gpu

```

You may notice that MXNet's NDArray is very similar to Numpy. One major difference is NDArray has a `context` attribute that specifies which device this array is on. In default it is `cpu()`. Now we change it to the first GPU. 

```{.python .input  n=4}
nd.zeros((3,4), ctx=gpu())
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "\n[[0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]]\n<NDArray 3x4 @gpu(0)>"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```
