This is the PyTorch implementation of our paper "ALNet: An Adaptive Attention Network with Local Discrepancy Perception for Accurate Indoor Visual Localization".

![overall](https://github.com/DAMMONGAO/alnet/assets/81282863/fbf01dd1-7a30-49ad-ba15-bb5744c4667f)

# video
Here is the video in a real scene:

https://github.com/DAMMONGAO/alnet/assets/81282863/fbf521e8-bd67-419b-ba14-90c1885cae52



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

# Datasets
We utilize two standard datasets (i.e, 7-Scenes and 12-Scenes) to evaluate our method.
- 7-Scenes: The 7-Scenes dataset can be downloaded from [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).
- 12-Scenes: The 12-Scenes dataset can be downloaded from [12-Scenes](https://graphics.stanford.edu/projects/reloc/).

# Evaluation
The pre-trained models can be downloaded from [7-Scenes](https://1drv.ms/u/s!AsLK4P4ia2R9biMdEyi_uQ-0No0?e=kLSPnh) and [12-Scenes](https://1drv.ms/u/s!AsLK4P4ia2R9bzafAEnZlrXiXsU?e=KF0AyW).
Then, we can run train_7S.sh or train_12S.sh to evaluate ALNet on 7-Scenes and 12-Scenes datasets. 
```bash
bash train_7S.sh
```
Notably, we need to modify the path of the downloaded models in train_7S.sh or train_12S.sh. 
The meaning of each part of train_7S.sh or train_12S.sh is as follows:
```bash
python main.py --model multi_scale_trans -dataset [7S/12S] --scene [scene name, such as chess] 
               --data_path ./data/ --flag test --resume [model_path]
```

# Training
We can run the train_7S.sh or train_12S.sh to train EAAINet.
The meaning of each part of train_7S.sh or train_12S.sh is as follows:
```bash
python main.py --model multi_scale_trans -dataset [7S/12S] --scene [scene name, such as chess] --data_path ./data/ 
               --flag train --n_epoch 500 --savemodel_path [save_path]
```
