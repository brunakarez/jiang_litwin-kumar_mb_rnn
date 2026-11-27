# jiang_litwin-kumar_mb_rnn

Implements the model of Jiang & Litwin-Kumar (2021)

## Setup

Install Conda (for example, Miniforge, Miniconda, or Anaconda), then clone or
download and extract this repository. Open a Conda-enabled terminal (such as
Anaconda Prompt on Windows), change to the repository root, and run:

```shell
conda env create -f environment.yml
conda activate jiang-litwin-kumar-mb-rnn
```

The environment file selects Python 3.12 and installs the pinned packages from
[`requirements.txt`](./requirements.txt). It is intended for Linux, macOS, and
Windows.

To open the notebook:

```shell
jupyter notebook test_notebook.ipynb
```

To run the training script instead:

```shell
python train.py
```
