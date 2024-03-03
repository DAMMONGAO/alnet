# import numpy as np
# from skimage.feature import greycomatrix, greycoprops
# from skimage import io, color
# import os
# import re

# def calculate_glcm_features(image):
#     # 将图像转换为灰度图
#     gray_image = color.rgb2gray(image)
    
#     # 将图像标准化为 8 位深度
#     gray_image = (gray_image * 255).astype('uint8')
    
#     # 计算 GLCM
#     glcm = greycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
    
#     # 计算 GLCM 特征
#     contrast = greycoprops(glcm, 'contrast')[0, 0]
#     homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
#     energy = greycoprops(glcm, 'energy')[0, 0]
#     entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

#     return contrast, homogeneity, energy, entropy

# def texture_score(contrast, homogeneity, energy, entropy):
#     # 计算纹理分数，优先考虑高对比度和低熵
#     score = contrast - entropy
#     return score

# def extract_number(filename):
#     # 提取文件名中的数字，用于排序
#     numbers = re.findall(r'\d+', filename)
#     return int(numbers[0]) if numbers else 0

# def process_folder(folder_path):
#     texture_scores = []

#     for file in os.listdir(folder_path):
#         if file.endswith('color.png'):
#             image_path = os.path.join(folder_path, file)
#             image = io.imread(image_path)

#             contrast, homogeneity, energy, entropy = calculate_glcm_features(image)

#             score = texture_score(contrast, homogeneity, energy, entropy)
#             texture_scores.append((file, score))

#     # 根据纹理分数排序，并平分为两组
#     texture_scores.sort(key=lambda x: x[1], reverse=True)
#     mid_index = len(texture_scores) // 2
#     clear_textures = [item[0] for item in texture_scores[:mid_index]]
#     repeated_textures = [item[0] for item in texture_scores[mid_index:]]

#     return sorted(clear_textures, key=extract_number), sorted(repeated_textures, key=extract_number)

# # 指定数据集路径
# dataset_path = '/home/xietao/dataset/LIVL/Floor5/no4'

# # 指定输出文件夹的基础路径
# output_base_path = '/home/jzq/texture_LIVL/floor5'

# # 创建输出文件夹
# os.makedirs(output_base_path, exist_ok=True)

# # 处理每个序列的图片
# for seq in ['RGB']:
#     clear_textures, repeated_textures = process_folder(os.path.join(dataset_path, seq))

#     # 将结果按序号排序并写入文件
#     with open(os.path.join(output_base_path, f'{seq}-qingxi.txt'), 'w') as f:
#         for item in clear_textures:
#             f.write(f"{item}\n")

#     with open(os.path.join(output_base_path, f'{seq}-mohu.txt'), 'w') as f:
#         for item in repeated_textures:
#             f.write(f"{item}\n")
            
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color
import os
import re

def calculate_glcm_features(image):
    # 将图像转换为灰度图
    gray_image = color.rgb2gray(image)
    
    # 将图像标准化为 8 位深度
    gray_image = (gray_image * 255).astype('uint8')
    
    # 计算 GLCM
    glcm = greycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
    
    # 计算 GLCM 特征
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

    return contrast, homogeneity, energy, entropy

def texture_score(contrast, homogeneity, energy, entropy):
    # 计算纹理分数，优先考虑高对比度和低熵
    score = contrast - entropy
    return score
def extract_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

def process_folder(folder_path):
    texture_scores = []

    for file in os.listdir(folder_path):
        if file.endswith('.color.png'):  # 修改为处理 png 结尾的文件
            image_path = os.path.join(folder_path, file)
            image = io.imread(image_path)

            contrast, homogeneity, energy, entropy = calculate_glcm_features(image)

            score = texture_score(contrast, homogeneity, energy, entropy)
            texture_scores.append((file, score))

    # 根据纹理分数排序，并平分为两组
    texture_scores.sort(key=lambda x: x[1], reverse=True)
    mid_index = len(texture_scores) // 2
    clear_textures = [item[0] for item in texture_scores[:mid_index]]
    repeated_textures = [item[0] for item in texture_scores[mid_index:]]

    return sorted(clear_textures, key=extract_number), sorted(repeated_textures, key=extract_number)

# 指定数据集路径
dataset_path = '/mnt/share/sda1/dataset/ghbdata/stairs/seq-04/'

# 指定输出文件夹的基础路径
output_base_path = '/mnt/share/sda1/dataset/ghbdata/mohustair4/'

# 创建输出文件夹
os.makedirs(output_base_path, exist_ok=True)

# 处理图片并输出结果
clear_textures, repeated_textures = process_folder(dataset_path)

# 将结果写入文件
with open(os.path.join(output_base_path, 'qingxi4.txt'), 'w') as f:
    for item in clear_textures:
        f.write(f"{item}\n")

with open(os.path.join(output_base_path, 'mohu4.txt'), 'w') as f:
    for item in repeated_textures:
        f.write(f"{item}\n")