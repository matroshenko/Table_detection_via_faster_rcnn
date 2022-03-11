# Table detection via Faster RCNN

This repository contains table detection algorithm based on this [Faster RCNN implementation](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN).

# Installation

TBD

# Data preparation

1. Download and extract [FinTabNet dataset](https://developer.ibm.com/exchanges/data/all/fintabnet/).
2. Run `python convert_pdfs_to_images.py /path/to/fintabnet/pdf /path/to/fintabnet/jpg`.
3. Set /path/to/fintabnet to variable _C.DATA.BASEDIR in config.py or alternatively set this variable during training/evaluation.

# Usage

TBD

# Results

All models were trained on fintabnet_train and evaluated on fintabnet_val datasets using
F1 score with IOU thresholds 0.5 and 0.75. All models were fine-tuned from ImageNet pretrained 
R50 models using NVidia GeForce GTX 1080 Ti.

| Model | F1@0.5 | F1@0.75 |
|-------|--------|---------|
| FPN FRCNN | 0.952 | 0.944 |

# Images

![Model predictions on random images from fintabnet_val](.github/fpn_predictions.png)