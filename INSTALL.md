# Installation

To install and use the 2DMatGMM package, you first need to install the required dependencies. This installation process has been tested using Python 3.10.

## Setting Up a Python Virtual Environment

For optimal compatibility, we recommend setting up a new Python virtual environment and installing the required packages within that environment. If you're using `conda` as your package manager, you can create and activate the virtual environment using the following commands:

```shell
conda create --name 2DMatGMM python=3.10
conda activate 2DMatGMM
```

## Installing Required Packages

Download the Repository and navigate to the repository directory and execute the following command:

```shell
pip install -e .
```

This will install all necessary packages and makes it possible to import the package with

```python
import GMMDetector
```

## Optional: Install Detectron2 for Evaluation

The Detectron2 package is not necessary for the basic use of 2DMatGMM, but it is required if you wish to perform model evaluations. To install Detectron2, execute the following commands:

```shell
pip install torch torchvision pillow==9.5.0
pip install git+https://github.com/facebookresearch/detectron2@67ac149
```

## Example Setup Using Conda

Here's an example of how to install the required packages using `conda` and set up a virtual environment for 2DMatGMM:

```shell
# setup virtual environment and install required packages
conda create --name 2DMatGMM python=3.10 -y
conda activate 2DMatGMM

# clone the repository and navigate to it
git clone https://github.com/Jaluus/2DMatGMM.git
cd 2DMatGMM

# install the package as GMMdetector
pip install -e .

# If you wish to perform model evaluations
pip install torch torchvision pillow==9.5.0
pip install git+https://github.com/facebookresearch/detectron2@67ac149  
```

This completes the setup process. Now, you're ready to use 2DMatGMM!
Check out the [Getting Started Guide](./GETTING_STARTED.md)!
