# A 60-min crash course on Gluon

This is a fork of mli/gluon-crash-course, customized for JSALT 2018 Summer School.

- [Deep learning session 1 slides](deep_learning_jsalt18.pdf)
- [Slides from guest speaker](http://www.cs.jhu.edu/~hmei/teach/gluon.pdf)

Every .md file in this repo may be run directly in Jupyter notebook when you use the `notedown` plugin. Keeping the format in markdown instead of .ipynb makes it easier to collaborate and review on GitHub.

## Prerequisites

### Install environment

1. Get [miniconda](https://conda.io/miniconda.html). (Skip if `conda` command is already available)
2. Set up Conda environment:
```
conda env create -f env/environment.yml
```


### Use notedown

#### Step 1: Install Jupyter notebook with notedown.

This tarball is provided to do this for you.

```
pip install https://github.com/mli/notedown/tarball/master
```

**Note**: If you run into a problem with this installation at the  `pandoc-attributes` step, run `pip uninstall pypandoc`, then try the installation tarball again.

#### Step 2: Start Jupyter with notedown

```
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

Before committing updates to these notebooks, please make sure all outputs are cleared.

## Use AWS

We provide the environment for JSALT 2018 Lab Session as public AMI: ami-e54d0f9d.
For detailed instructions, see [Run on AWS](use_aws.md)
