# Install a different MXNet package

In this tutorial, we show how to install a different MXNet version through
`pip`. The plain `mxnet` which can be installed by `pip install mxnet`, is able
to execute almost all MXNet codes in CPUs. But there are other pre-compiled
packages to support more hardware and/or more efficient executions.

## Uninstall a previously installed version

To install a different version, we should uninstall the previously installed
version first. We can check it through

```bash
pip list | grep mxnet
```

If the previous command returns a non-empty result, such as `mxnet
(1.2.0b20180413)`, then we can remove it by

```bash
pip uninstall mxnet
```

## Choose another package

All MXNet packages can be found at
[pypi.org](https://pypi.org/search/?q=mxnet). Here we list three major variants.

### Nvidia GPUs

To run on Nvidia GPUs, we should install a package with `cu??` in the package
name, where `??` is CUDA version such as `80` and `91`. To install such as
version, users should have [CUDA](https://developer.nvidia.com/cuda-downloads)
installed first. Then according to the CUDA version, which can be checked by
`nvcc --version`, to select the proper package.

```eval_rst

============  ===============
CUDA version  MXNet package
============  ===============
7.5           ``mxnet-cu75``
8.0           ``mxnet-cu80``
9.0           ``mxnet-cu90``
9.1           ``mxnet-cu91``
============  ===============

```

All `cu` packages ship `cudnn` in default, there is no need to install it
separately.

A common error to use the `cu` packages is failed to open CUDA shared objects
after `import mxnet`, such as

```
OSError: libcudart.so.9.0: cannot open shared object file: No such file or directory
```

To solve, we just need to add CUDA into the library path, such as on Linux, we can run

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
```

### Intel CPUs

All MXNet packages support Intel CPUs, but the variants with `mkl` in the
package name can potentially improve the performance. For example, for
convolutional neural network inference, `mxnet-mkl` often outperforms `mxnet` by
[more than 4x](https://mxnet.incubator.apache.org/faq/perf.html#intel-cpu).

But also note that `mxnet-mkl` is still experimental. You may need to roll back
to `mxnet` if your programs are failed to run.

### Nvidia GPU + Intel CPUs

We can have both hardware accelerated:


```eval_rst

============  ==================
CUDA version  MXNet package
============  ==================
7.5           ``mxnet-cu75mkl``
8.0           ``mxnet-cu80mkl``
9.0           ``mxnet-cu90mkl``
9.1           ``mxnet-cu91mkl``
============  ==================

```

## Upgrade to the newest version

MXNet often makes a major release in one or two months. In addition, it releases
nightly builds every day. Some toolkits or tutorials require the newest version,
which can be installed or upgraded through the `--pre` flag.

Install the nightly build MXNet with CUDA 9.1:

```bash
pip install --pre mxnet-cu91
```

or upgrade the version with `-U`:

```bash
pip install --pre -U mxnet-cu91
```


## Other installation options

If you want to install MXNet in a different way, such as with Scala frontend or
Docker, refer to
[MXNet installation](http://mxnet.incubator.apache.org/install/index.html) for
more details.
