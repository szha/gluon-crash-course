# Manipulate data with `ndarray`

We’ll start by introducing `NDArray`s, MXNet’s primary tool for storing and transforming data. If you’ve worked with `NumPy` before, you’ll notice that `NDArray`s are, by design, similar to `NumPy`’s multi-dimensional array.

## Get started

To get started, let's import the `ndarray` package (shorten for `nd`) from MXNet.

```{.python .input  n=1}
from mxnet import nd
```

Next, let's see how to create a 2D array (also called matrix) with values from a tuple of int tuple.

```{.python .input  n=2}
nd.array(((1,2,3),(5,6,7)))
```

We can also create a matrix with the same shape (2 rows and 3 columns) but values are initialized by 1s.

```{.python .input  n=3}
x = nd.ones((2,3))
x
```

Often, we’ll want to create arrays whose values are sampled randomly. For example, sampling values uniformly between -1 and 1.

```{.python .input  n=4}
y = nd.random.uniform(-1,1,(2,3))
y
```

As in NumPy, the dimensions of each NDArray are accessible via the `.shape` attribute. We can also query its `size`, which is equal to the product of the components of the shape. In addtion, `.dtype` tells the data type of the stored values.

```{.python .input  n=5}
(x.shape, x.size, x.dtype)
```

## Operations

NDArray supports a large number of standard mathematical operations. Such as element-wise multiplication:

```{.python .input  n=6}
x * y
```

exponentiation:

```{.python .input  n=7}
y.exp()
```

And grab a matrix’s transpose to compute a proper matrix-matrix product.

```{.python .input  n=8}
nd.dot(x, y.T)
```

## Indexing

MXNet NDArrays support slicing in all the ridiculous ways you might imagine accessing your data. Here’s an example of reading a particular element, which returns a 1D array with shape `(1,)`

```{.python .input  n=9}
y[1,2]
```

Read the second and third columns from `y`.

```{.python .input  n=10}
y[:,1:3]
```

and writing to a specific element

```{.python .input  n=11}
y[:,1:3] = 2
y
```

Multi-dimensional slicing is also supported.

```{.python .input  n=12}
y[1:2,0:2] = 4
y
```

## Converting between MXNet NDArray and NumPy

Converting MXNet NDArrays to and from NumPy is easy. The converted arrays do not share memory.

```{.python .input  n=13}
a = x.asnumpy()
(type(a), a)
```

```{.python .input  n=14}
nd.array(a)
```
