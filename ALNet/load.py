import numpy as np


# 加载保存的数组
loaded_data = np.load("chess.npy")

# 打印加载的数组
print(loaded_data)
print(loaded_data.shape)