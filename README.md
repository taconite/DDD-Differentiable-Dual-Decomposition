# Differentiable Dual-Decomposition for Semantic Image Segmentation

## Introduction
Pytorch code for our paper [*End-to-end Training of CNN-CRF via Differentiable Dual-Decomposition*](https://arxiv.org/abs/1912.02937).

## Overview
- `image_segmentation/`: includes training and validation scripts.
- `lib/`: contains core functions, data preparation, custom layers, model definition, and utility functions.
- `experiments/`: contains `*.yaml` configuration files to run experiments.

## Requirements
The code is developed using python 3.7.2 on Ubuntu 18.04.1. NVIDIA GPUs ared needed to train and test. 
See [`requirements.txt`](requirements.txt) for other dependencies.

## Quick start
### Installation
1. Install pytorch == v1.0.0 with CUDA>=9 following [official instructions](https://pytorch.org/).
2. Clone this repo, and we will call the directory that you cloned as `${ROOT}`
3. Install dependencies.
   ```
   pip install -r requirements.txt
   ```
4. Add current project directory (which we will later denote as ${DDD_ROOT}) to PYTHONPATH environment variable.
   ```
   export PYTHONPATH=${PYTHONPATH}:${PWD}
   ```
4. Compile the custom layers in ./lib/layers/dp-extension and ./lib/models/sync_bn/inplace_abn/src/ by going into these directories and running the following command:
   ```
   python setup.py install
   ```

### Data Preparation for PASCAL VOC 2012 benchmark
1. Download original [dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) of PASCAL VOC2012 and extract the tarball file onto your disk, we denote the location where dataset was extracted as ${VOC_ROOT}. It should have a folder called `VOCdevkit`.
2. Create a new directory named `data` at root directory of this project.
3. Create a symbolic link to VOC2012 dataset via following command: 
   ```
   ln -s ${VOC_ROOT}/VOCdevkit/VOC2012/ ${DDD_ROOT}/data/pascal_voc
   ```
4. After all above steps you should have the following structure in ./data/pascal_voc:
   ```
   ${DDD_ROOT}
    `-- data
        `-- pascal_voc 
            |-- Annotations
            |-- ImageSets
            |-- JPEGImages
            |-- SegmentationClass
            `-- SegmentationObject

   ```
5. Download the [Berkley augmented dataset](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz) for additional annotations on VOC2012. Extract the .tgz file to an arbitrary location which we will denote as ${SBD_ROOT}$, it should have a folder called `benchmark_RELEASE`.
6. Create symbolic link to SBD dataset via following command:
   ```
   ln -s ${SBD_ROOT}/benchmark_RELEASE/ ${DDD_ROOT}/data/sbd
   ```

### Training on VOC2012
1. Download ImageNet-pretrained ResNet-50 [model](https://download.pytorch.org/models/resnet50-19c8e357.pth) and put it under `${ROOT}/models/pytorch/imagenet/`
2. To train baseline DeepLabV3 model (with ASPP), run:
```
CUDA_VISIBLE_DEVICES=$GPU_IDS python image_segmentation/train_voc.py --cfg experiments/pascal_voc/resnet50-aspp_513x513_head-lr-1x_sgd-poly-lr7e-3_2gpus.yaml
```
To train DeepLabV3 with dual-decomposition end-to-end, run:
```
CUDA_VISIBLE_DEVICES=$GPU_IDS python image_segmentation/train_voc.py --cfg experiments/pascal_voc/resnet50-aspp_513x513_ne-fpi-iter5_head-lr-1x_sgd-poly_lr7e-3_2gpus.yaml
```
Two GPUs each with >=11GB memory are required for training either of the models.
Model checkpoints and logs will be saved into `output` folder while tensorboard logs will be saved into `log` folder.

### Testing the model
To test the model after training, run:
```
CUDA_VISIBLE_DEVICES=$GPU_ID python image_segmentation/validate_voc.py --cfg ${PATH_TO_CONFIG_FILE}
```
Where ${PATH_TO_CONFIG_FILE} is the same file used in training. Tensorboard logs will be saved into `log` folder.

## Citation
If you use our code or models in your research, please cite with:
```
@article{wang2019DDD,
    author  = {Shaofei Wang and Vishnu Lokhande and Maneesh Singh and Konrad Kording and Julian Yarkony},
    title   = {End-to-end Training of CNN-CRF via Differentiable Dual-Decomposition},
    journal = {CoRR},
    volume  = {abs/1912.02937},
    year    = {2019},
    url     = {http://arxiv.org/abs/1912.02937}
}
```

## References
- The overall structure of the code follows [Simple Baselines for Human Pose Estimation and Tracking](https://github.com/microsoft/human-pose-estimation.pytorch).
- Xception back-bone definition and weights are from [Pretrained models for Pytorch](https://github.com/Cadene/pretrained-models.pytorch).
