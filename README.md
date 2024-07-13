# MTL-Split: Multi-Task Learning for Edge Devices using Split Computing #

## Installation ##
**1. Repository setup:**
* `$ git clone https://github.com/intelligolabs/MTL-Split`
* `$ cd MTL-Split`
* Download the 3D Shapes dataset from [Google Cloud Storage](https://console.cloud.google.com/storage/browser/3d-shapes;tab=objects?prefix=&forceOnObjectsSortingFiltering=false). Then, add the dataset path on line 27 of the file `shapes.py`
* Download the MEDIC dataset from https://crisisnlp.qcri.org/medic/
* Download the FACES dataset from https://faces.mpdl.mpg.de/imeji/

**2. Conda environment setup:**
* `$ conda create -n mtl_split python=3.10`
* `$ conda activate mtl_split`
* `$ pip install -r requirements.txt`

Optionally, you can also log the training and evaluation to [wandb](https://wandb.ai).
* Update line 102 of the file `main.py`, specifying `project=''` and `entity=''`

## Run MTL-Split ##
To run MTL-Split, use the file `main.py`.
In particular, the `launch.sh` file it contains two example (STL & MTL) of a launch script exmaple that you can use to modify the default configuration.

## Authors ##
Luigi Capogrosso<sup>1</sup>, Enrico Fraccaroli<sup>1,2</sup>, Samarjit Chakraborty<sup>2</sup>, Franco Fummi<sup>1</sup>, Marco Cristani<sup>1</sup>

<sup>1</sup> *Department of Engineering for Innovation Medicine, University of Verona, Italy*

<sup>2</sup> *Department of Computer Science, The University of North Carolina at Chapel Hill, USA*

<sup>1</sup> `name.surname@univr.it`, <sup>2</sup> `enrifrac@cs.unc.edu`, `samarjit@cs.unc.edu`

## Citation ##
If you use [**MTL-Split**](https://arxiv.org/abs/2407.05982), please, cite the following paper:
```
@article{capogrosso2024mtl,
  title={MTL-Split: Multi-Task Learning for Edge Devices using Split Computing},
  author={Capogrosso, Luigi and Fraccaroli, Enrico and Chakraborty, Samarjit and Fummi, Franco and Cristani, Marco},
  journal={arXiv preprint arXiv:2407.05982},
  year={2024}
}
```
