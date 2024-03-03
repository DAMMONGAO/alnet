This is the PyTorch implementation of our paper "ALNet: An Adaptive Attention Network with Local Discrepancy Perception for Accurate Indoor Visual Localization".

![overall](https://github.com/DAMMONGAO/alnet/tree/main/ALNet/assets)

# video
(https://github.com/DAMMONGAO/alnet/issues/1#issue-2165325963)


# Installation
Firstly, we need to set up python3 environment from requirement.txt:

```bash

https://github.com/DAMMONGAO/alnet/assets/81282863/d074e1a0-5dfc-42ba-a19d-aaa4d87a851e


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
The pre-trained models can be downloaded from [7-Scenes](https://1drv.ms/u/s!AsLK4P4ia2R9biMdEyi_uQ-0No0?e=kLSPnh) ansd [12-Scenes](https://1drv.ms/u/s!AsLK4P4ia2R9bzafAEnZlrXiXsU?e=KF0AyW).
Then, we can run tran_7S.sh or train_12S.sh to evaluate EAAINet on 7-Scenes and 12-Scenes datasets. 
```bash
bash tran_7S.sh
```
Notably, we need to modify the path of the downloaded models in tran_7S.sh or train_12S.sh. 
The meaning of each part of tran_7S.sh or train_12S.sh is as follows:
```bash
python main.py --model multi_scale_trans -dataset [7S/12S] --scene [scene name, such as chess] 
               --data_path ./data/ --flag test --resume [model_path]
```

# Training
We can run the tran_7S.sh or train_12S.sh to train EAAINet.
The meaning of each part of tran_7S.sh or train_12S.sh is as follows:
```bash
python main.py --model multi_scale_trans -dataset [7S/12S] --scene [scene name, such as chess] --data_path ./data/ 
               --flag train --n_epoch 500 --savemodel_path [save_path]
```
