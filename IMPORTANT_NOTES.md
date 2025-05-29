### Rename the root folder from vapo_affordance_model_only to vapo_aff

This branch is under testing and development, thus the package should not be installed yet.
Create a venv named venv_vapo inside vapo_aff.

Use the following every time a new command line is opened:

`
source venv_vapo/bin/activate
`

`
export PYTHONPATH=/home/admi/vapo_aff
`

*replace /home/admi with your path to the folder.

**run all scripts from the root folder, outside /scripts.**
**otherwise, yaml config paths will not work**




## Install notes:

Before anything, make sure the nvidia drivers, smi and cuda toolkit are installed and working. Install Eigen, then the hough_vouting, and for last run `python setup.py install`