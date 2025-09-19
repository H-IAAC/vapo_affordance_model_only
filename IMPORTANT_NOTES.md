### First of all, for ease of use, rename the root folder from vapo_affordance_model_only to vapo_aff

## Install notes:

Before starting, make sure the nvidia drivers, smi and cuda toolkit are installed and working. Then, create a venv named venv_vapo inside vapo_aff. Install the requirements with "pip install -r requirements.txt".

After that, install Eigen, then the hough_vouting, following the instructions on the README at the "Install the Hough voting layer" section. For last run `python setup.py install`.


## General usage

Use the following every time a new command line is opened:

`
source venv_vapo/bin/activate
`

In case you did not run setup.py or do not want to install this as a package, you can use the following:
`
export PYTHONPATH=/[...]/vapo_aff
`

*replace [...] with your path to the folder.

After that, you are good to go. But always **run scripts from the vapo_aff folder, outside /scripts. Otherwise, yaml config paths will not work.**
The main script is the scripts/viz_affordances.py and you should be able to run it with `python ./scripts/viz_affordances.py data_dir=datasets/playdata/viz_affordances/input_files/`


## Using data

Inside input_files folder, there must be <cam_type>/<file_type>/<files.ext> as in static/rgb/0001.png. Take a look at the exemples provided. Notice that the rgb and depth files must be paired (that is, a rgbd image split between rgb and d, with the same file name).
