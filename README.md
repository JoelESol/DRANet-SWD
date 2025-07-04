# DRANet-SWD — SMC 2025
### An Improvement to [DRANet](https://github.com/Seung-Hun-Lee/DRANet) using [Sliced Wasserstein Discrepancy](https://github.com/apple/ml-cvpr2019-swd)
### 
## Requirements
```
Pytorch 2.6.0
CUDA 11.8
python 3.12
numpy 2.1.2
scipy 1.15.2
tensorboardX
prettytable
```
## Data Preparation
Download [MNIST-M](https://github.com/fungtion/DANN), [Cityscapes](https://www.cityscapes-dataset.com/), [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/)
## Folder Structure of Datasets
```
├── data
      ├── MNIST
      ├── USPS
      ├── mnist_m
            ├── mnist_m_train
                      ├── *.png
            ├── mnist_m_test
                      ├── *.png
            ├── mnist_m_train_labels.txt
            ├── mnist_m_test_labels.txt
      ├── Cityscapes
            ├── GT
                   ├── train
                   ├── val
                   ├── test
            ├── Images
                   ├── train
                   ├── val
                   ├── test
      ├── GTA5
            ├── GT
                   ├── 01_labels
                   ├── 02_labels
                   ├── ...
            ├── Images
                   ├── 01_images
                   ├── 02_images
                   ├── ...
      
├── data_list
      ├── Cityscapes
              ├── train_imgs.txt
              ├── val_imgs.txt
              ├── train_labels.txt
              ├── val_labels.txt
      ├── GTA5
              ├── train_imgs.txt
              ├── train_labels.txt

```
## Train
You must input the task(clf or seg), style (gram, swd), datasets(M, MM, U, G, C) and experiment name.
```
python train.py -T [task] --style [style] -D [datasets] --ex [experiment_name]

Example:  python train.py -T clf --style swd -D M MM --ex M2MM_swd
```
## Test
Input the same experiment_name that you trained and specific iteration.
```
python test.py -T [task] -D [datasets] --ex [experiment_name (that you trained)] --load_step [specific iteration]
Example:  python test.py -T clf -D M MM --ex M2MM --load_step 100000
```
## Tensorboard
You can see all the results of each experiment on tensorboard.
```
CUDA_VISIBLE_DEVICES=-1 tensorboard --logdir tensorboard --bind_all
```
