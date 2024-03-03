import numpy as np

# 创建一个维度为（4，64，1200）的示例数组
array = np.random.rand(4, 3, 4)
# print(array)

###欧式距离

# 欧氏距离
def compute_euclidean_similarity(batch):
    """
    计算一个batch内部的点之间的相似度矩阵
    :param batch: 输入数组，形状为(batch_size, num_channels, num_points)
    :return: 相似度矩阵，形状为(batch_size, num_points, num_points)
    """
    dists = euclidean_distance(batch)
    sim = 1 / (1 + dists)  # 取欧式距离的倒数得到相似度
    return sim
def euclidean_distance(batch):
    """
    计算一个batch内部的点之间的欧式距离
    :param batch: 输入数组，形状为(batch_size, num_channels, num_points)
    :return: 欧式距离矩阵，形状为(batch_size, num_points, num_points)
    """
    batch_size, num_channels, num_points = batch.shape
    batch_reshaped = batch.reshape(batch_size, -1, num_channels)  # 将每个batch展平为(batch_size, num_points, num_channels)
    dists = np.sqrt(np.sum((batch_reshaped[:, :, np.newaxis, :] - batch_reshaped[:, np.newaxis, :, :]) ** 2, axis=-1))
    return dists

##曼哈顿距离 最大为1 设置0
def compute_manhattan_similarity(batch):
    """
    计算一个batch内部的点之间的相似度矩阵
    :param batch: 输入数组，形状为(batch_size, num_channels, num_points)
    :return: 相似度矩阵，形状为(batch_size, num_points, num_points)
    """
    dists = manhattan_distance(batch)
    sim = 1 / (1 + dists)  # 取欧式距离的倒数得到相似度
    return sim
def manhattan_distance(batch):
    """
    计算一个batch内部的点之间的曼哈顿距离作为相似度
    :param batch: 输入数组，形状为(batch_size, num_channels, num_points)
    :return: 曼哈顿相似度矩阵，形状为(batch_size, num_points, num_points)
    """
    batch_size, num_channels, num_points = batch.shape
    batch_reshaped = batch.reshape(batch_size, num_channels, -1)  # 将每个batch展平为(batch_size, num_channels, num_points)
    # 计算曼哈顿距离
    distances = np.abs(batch_reshaped[:, :, :, np.newaxis] - batch_reshaped[:, :, np.newaxis, :])
    manhattan_distances = np.sum(distances, axis=1)  # 沿着通道维度求和，得到曼哈顿距离
    return manhattan_distances
    
# 皮尔逊相关性 设置负无穷

def compute_pearson_similarity(batch):
    """
    计算一个batch内部的点之间的皮尔逊相似度
    :param batch: 输入数组，形状为(batch_size, num_channels, num_points)
    :return: 皮尔逊相似度矩阵，形状为(batch_size, num_points, num_points)
    """
    batch_size, num_channels, num_points = batch.shape
    batch_reshaped = batch.reshape(batch_size, -1, num_channels)  # 将每个batch展平为(batch_size, num_channels*num_points)
    similarities = np.zeros((batch_size, num_points, num_points))
    for i in range(batch_size):
        similarities[i] = np.corrcoef(batch_reshaped[i])
    return similarities

#余弦相似度

def compute_cosine_similarity(batch):
    """
    计算一个batch内部的点之间的余弦相似度
    :param batch: 输入数组，形状为(batch_size, num_channels, num_points)
    :return: 余弦相似度矩阵，形状为(batch_size, num_points, num_points)
    """
    batch_size, num_channels, num_points = batch.shape
    batch_reshaped = batch.reshape(batch_size, -1, num_channels)  #
    # 计算每个点的模
    norms = np.linalg.norm(batch_reshaped, axis=-1, keepdims=True)
    # 计算点积
    dot_products = np.einsum('ijk,ihk->ijh', batch_reshaped, batch_reshaped)
    # 计算余弦相似度
    similarities = dot_products / (norms * norms.transpose(0, 2, 1))
    return similarities


# 计算相似度矩阵
#similarity = compute_euclidean_similarity(array) #欧式距离
#similarity = compute_manhattan_similarity(array) #曼哈顿距离
#similarity = compute_pearson_similarity(array) #皮尔逊相关系数
similarity = compute_cosine_similarity(array) #余弦相似度

print(similarity)
print(similarity.shape)