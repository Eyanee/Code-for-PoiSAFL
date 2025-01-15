# Code For PoiSAFL

##  About the Project
This is the code repository for the paper PoiSAFL. The "PoiSAFL_Code" folder contains the source code. File prefixed with "Async" are the main function files. The "otherGroupingMethods.py" and "otherPoisoningMethods.py" files implement baseline defense and attack methods. Files prefixed with "poison_optimization" are the specific implementation code for the PoiSAFL method. The remaining files are implementaions of models, datasets and other functional codes. Additionally, the "scripts" folder provides two example launch scripts. You can modify the parameters in these scripts to conduct various experiments, with parameter settings referenced from the "option.py" file.

## Getting Started
Follow these steps to run this code locally.

### Prerequisites
This code was tested on the following environments:

* Ubuntu 18.04
* Python 3.7
* PyTorch 1.12.0
* CUDA 11.6

### Steps to Run the Code
* get into folder `scripts`
* adjust the code parameters according to the required running scenario.
* All parameters required for the experiment are described in file `options.py.` Please see the python file for a detailed description of the parameters.

````
# for PoiSAFL attack  under normal settings
 bash PoiSAFL_test.sh

# for other baseline attack under normal settings
 bash PoiSAFL_other.sh

````
