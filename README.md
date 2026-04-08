Code for TCMMAC training

## Requirements

The environment for running this code has been tested and verified with the following versions:

* python 3.10.19
* pytorch 2.4.0
* cuda 11.8

## Training

To train the model, run the following script:

* bash ./scripts/run_`target dataset`.sh

Replace `<target_dataset>` with the name of the desired dataset (e.g., `cifar10/100`, `dvs_cifar10`, or `imgnet`).

## Dataset Structure

The expected directory structure for the datasets is as follows:

```
TCM-MAC/dataset/
├──CIFAR/
│	├──train/
│	├──val/
├──DVS_CIFAR10/
│	├──events_np/
│	├──extract/
│	├──frames_number_10_split_by_number/
├──ImageNet/
│	├──train/
│	├──val/
├── ......
```