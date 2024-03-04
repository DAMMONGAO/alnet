This is the PyTorch implementation of our paper "ALNet: An Adaptive Attention Network with Local Discrepancy Perception for Accurate Indoor Visual Localization".

![overall](https://github.com/DAMMONGAO/alnet/assets/81282863/fbf01dd1-7a30-49ad-ba15-bb5744c4667f)

# video
We make videos of the results of the experiments. The RGB image is on the left side of the video screen, the predicted trajectory and ground truth trajectory are in the middle, and the uncertainty is on the right.
Here is the video in a real scene:

https://github.com/DAMMONGAO/alnet/assets/81282863/fbf521e8-bd67-419b-ba14-90c1885cae52

Here is the video in Pumpkin of 7Scenes Dataset

https://github.com/DAMMONGAO/alnet/assets/81282863/da698652-b37c-45c3-a07f-7dd37226e306

# Installation
Firstly, we need to set up python3 environment from requirement.txt:

```bash

pip3 install -r requirement.txt 
```

Subsequently, we need to build the cython module to install the PnP solver:
```bash
cd ./pnpransac
rm -rf build
python setup.py build_ext --inplace
```
Take 7-Scenes Dataset as example:
# Dataset
The dataset can be downloaded from [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).

# Evaluation
The pre-trained models can be downloaded from [7-Scenes](https://1drv.ms/f/s!Aln-nNBY3wXyaK5NNluyMMf-WJo?e=vpX9cb)
Then, we can run train_7S.sh to evaluate ALNet.
```bash
bash train_7S.sh
```
Notably, we need to modify the path of the downloaded models in train_7S.sh. 
The meaning of each part of train_7S.sh is as follows:
```bash
python main.py --model multi_scale_trans -dataset [7S] --scene [scene name, such as chess] 
               --data_path ./data/ --flag test --resume [model_path]
```

# Training
We can run the train_7S.sh to train ALNet.
The meaning of each part of train_7S.sh is as follows:
```bash
python main.py --model multi_scale_trans -dataset [7S] --scene [scene name, such as chess] --data_path ./data/ 
               --flag train --n_epoch 500 --savemodel_path [save_path]
(12Scenes is the same.)
```
