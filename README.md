# CloudNet: Deep Neural Network for Relocalization in Point Clouds Maps
This is the PyTorch implementation for CloudNet, a neural network realized during a research internship. The code is based on the [PoseLSTM](https://github.com/hazirbas/poselstm-pytorch) code.

## Prerequisites
- Linux
- Python 3.5.2
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Clone this repo:
```bash
git clone https://github.com/aRI0U/cloudnet.git
python3 -m pip install -r requirements.txt
```

### Train a model

- Download a dataset composed of:
  - a directory containing point clouds
  - a text file containing the poses of each point cloud

- Train a model
```bash
python3 train.py --dataroot <path_of_your_dataset>
```

You can add several options to choose the number of points per point cloud, the model used (PointNet based or PointCNN based), how to split the dataset, etc.

Each run of `train.py` creates a new experiment whose checkpoints are stored into folder `checkpoints/<name_of_experiment>`. You can specify the name of the experiment with option `--name`. Otherwise the name of the experiment is by default `<model>/<datetime>`.

#### Continue training

You can interrupt the training of a model and continue it later. In order to do so, type the following:
```bash
python3 train.py --continue --name <name>
```
where `<name>` is the name of the experiment you want to keep on training. Options of the previous training are automatically loaded, so you do not have to specify again the model, number of points per point cloud, etc.

### Test a model

- Evaluate a specific model
```bash
python3 test.py --name <name>
```

Also here, the options provided during training phase are stored so that you do not have to specify them again.

#### Splitting dataset

Option `--split <n>` during training splits your dataset into a training set and a validation set by loading only one frame over `<n>`.

Then, during test phase, add `--phase val` to test your model only on the validation set or `--phase retrain` to test only on your training set. By default, testing is done on the whole dataset.

## Citation
```
@inproceedings{PoseNet15,
  title={PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization},
  author={Alex Kendall, Matthew Grimes and Roberto Cipolla },
  journal={ICCV},
  year={2015}
}
@inproceedings{PoseLSTM17,
  author = {Florian Walch and Caner Hazirbas and Laura Leal-Taixé and Torsten Sattler and Sebastian Hilsenbeck and Daniel Cremers},
  title = {Image-based localization using LSTMs for structured feature correlation},
  month = {October},
  year = {2017},
  booktitle = {ICCV},
  eprint = {1611.07890},
  url = {https://github.com/NavVisResearch/NavVis-Indoor-Dataset},
}
@article{PointNet,
title = {{PointNet: Deep learning on point sets for 3D classification and segmentation}},
author = {Qi, Charles R. and Su, Hao and Mo, Kaichun and Guibas, Leonidas J.},
journal = {Proceedings - 30th IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017},
year = {2017}
doi = {10.1109/CVPR.2017.16},
eprint = {1612.00593},
url = {http://arxiv.org/abs/1612.00593},
}
@article{PointNet++,
title = {{PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space}},
author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
journal = {CoRR},
year = {2017}
eprint = {1706.02413},
url = {http://arxiv.org/abs/1706.02413},
}
@article{PointCNN,
title = {{PointCNN: Convolution On {\$}\backslashmathcal{\{}X{\}}{\$}-Transformed Points}},
author = {Li, Yangyan and Bu, Rui and Sun, Mingchao and Wu, Wei and Di, Xinhan and Chen, Baoquan},
journal = {CoRR},
year = {2018}
eprint = {1801.07791},
url = {http://arxiv.org/abs/1801.07791},
}
```
## Acknowledgments
Code is inspired by [poselstm-pytorch](https://github.com/hazirbas/poselstm-pytorch).
