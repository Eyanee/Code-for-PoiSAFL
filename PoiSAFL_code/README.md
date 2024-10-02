# Code For PoiSAFL

##  About the Project
This is the code repository for the paper PoiSAFL.

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