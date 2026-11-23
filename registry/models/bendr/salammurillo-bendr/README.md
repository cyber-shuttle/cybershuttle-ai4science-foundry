# DN3 Environment & Setup

## Initial Setup
BENDR is an extension of its underlying framework, DN3. 
As of the latest updates, BENDR is intregrated into the DN3 Repository.

The repositoy is found here
https://github.com/SPOClab-ca/dn3/

Start setting up by cloning the repository

```bash
git clone https://github.com/SPOClab-ca/dn3.git && cd dn3
```

There are a few files that need modification due to outdated code

DN3 was created on Python 3.7 so some imports have changed. Python 3.10 is the minimum requirement to run on the CUDA 13 GPUs ICE uses

in dn3/dn3/utils.py, the collections import is updated to collections.abc

in dn3/dn3/configuratron/config.py, the arguments passed to the function are corrected to match those the model retrieves by parsing the configuration YML file

in dn3/examples/basics.ipynb, the workflow code is also corrected to fix simple TypeErrors

The corrected files are all provided in this folder, replace the files in the above mentioned paths with these sharing the same name.

## ICE Setup

DN3 was set up using 
- 4 CPU Cores
- Nvdia H100 HGX
- 16GB RAM

An example slurm script to run with this hardware is

```bash
salloc --ntasks=1 --cpus-per-task=4 --mem=16G --gres=gpu:H100:1 --time=01:00:00
```


## Create the environment

The exact environment export is stored in
[`environment-dn3-legacy.yml`](./environment-dn3-legacy.yml). 

Create the Conda environment from the recorded export:

```bash
module load anaconda3
conda env create -f environment.yml
conda activate dn3_env
```

And then use pip to install other requirments and dn3

```bash
python -m pip install -r requirements.txt
python -m pip install -e ./dn3
```

Using the exported conda environment is strongly recommended as dn3 requires 
very specific versions of its required packages in order for all dependencies to
be compatible.

Initial setup was complicated as DN3 was made with Python 3.7 and many dependencies
require older versions for the models to run on ICE.

Of these complications was a five way conflict between CUDA, pyyaml, pytorch, numpy, moabb.
The compatible versions are all pinned in the environment above.

The CUDA version ICE uses requires Python 3.10+ which 


## Main dependencies

The recorded environment includes the following versions used by this setup:

| Package | Version |
| --- | --- |
| Python | 3.11.17 |
| DN3 | 0.2a0 |
| PyTorch | 2.13.0+cu132 |
| MNE | 1.5.1 |
| NumPy | 1.23.5 |
| pandas | 1.5.3 |
| PyYAML | 5.3.1 |
| pyyaml-include | 1.2 |
| tqdm | 4.70.1 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.9 |
| Jupyter/IPython | 7.3.0 / 9.17.1 |

The provided requirements.txt fron DN3 did not actually include all the required packages.
Many packages imported in the code, such as moabb, were not included in the provided list.
The environemnt above contains all those that were not originally included.

## Verify the environment

Run the following after installation:

```bash
python - <<'PY'
import sys

import dn3
import mne
import numpy
import pandas
import torch
import yaml

print("Python:", sys.version.split()[0])
print("DN3:", dn3.__file__)
print("MNE:", mne.__version__)
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyYAML:", yaml.__version__)
PY
```

For this setup, `torch.cuda.is_available()` should be `True`.
CPU execution is very slow and not recommended.

# Inference Workflow

The authors of DN3 generously provide an example inference using the PhysioNet
EEG Motor/Movement Imagery Dataset

https://physionet.org/content/eegmmidb/1.0.0/

The example inference classifies EEG recordings, predicting whether the EEG recordings are hand or foot movements (T1 or T2).
The model is trained on the above dataset, which contains EEG recordings from 109 Participants.

## Retreving Dataset

The best way to retrieve the dataset is downloading in using AWS CLI and transfering using Globus.
The PhysioNet server is very slow and retreiving the dataset without AWS can take hours.

Set up AWS CLI on your local machine if you do not have it following these instructions

https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html

Once Setup use the following to retreive the dataset, while specifying a path to the desired destination

```bash
aws s3 sync --no-sign-request s3://physionet-open/eegmmidb/1.0.0/ DESTINATION
```

Then transfer to ICE using Globus, which PACE provides instructions here

https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0041890

## Running Training and Inference

The basics notebook contains the workflow to run the model training and inference.

The configuration to run the training and inference off the PhysioNet dataset is provided in
[`my_config.yml`](./my_config.yml).

Put the path to the dataset in this configuraiton file where it says toplevel

The other fields are already filled out to the specifications of the example.

In the Jupyter Notebook, select the dn3_env as the kernel and run all cells.
This should construct the model and run the training and inference.

The example divides the data into five folds, with 60% of participants used for training, 20% used for validation, and 20% for testing. 
First the model learns from the training group, then classifies the EEG as hand movement (T1) or foot movement (T2) and compares its prediction to the known label.
Then the model with the best accuracy and smallest validation lost is chosen and tested on the remaining testing group to provide final validation and accuracy.
The results of the predictions and validation checks are present in the exported notebook below.

The example runs tests, over all five folds, and per participant.

The accuracy and full results are output in the notebook, and a successful execution was exported and stored at
[`basics.html`](./basics.html).

## Model an Weights

The specific model used in TIDNet which is defined in dn3/trainable/models.py

The Weights for this example are kept in memory, and are not stored.

For BENDR and downstream tasks, the authors use a much larger dataset and the pretrained weights can be found here
https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha
