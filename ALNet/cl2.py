import re

def transform_filename(filename):
    # 使用正则表达式提取数字部分
    match = re.match(r"frame-(\d+)\.color\.png", filename)
    if match:
        # 提取到的数字部分
        frame_number = match.group(1)
        # 生成新的文件名
        new_filename = f"stairs seq-04 frame-{frame_number}"
        return new_filename
    else:
        # 如果文件名格式不匹配，则返回原始文件名
        return filename

# 读取txt文件中的数据
with open("/mnt/share/sda1/dataset/ghbdata/mohustair4/mohu4.txt", "r") as file:
    # 读取所有行
    lines = file.readlines()

# 将转换后的数据写回原始文件
with open("/mnt/share/sda1/dataset/ghbdata/mohustair4/mohu4.txt", "w") as file:
    for line in lines:
        # 移除换行符
        line = line.strip()
        # 转换文件名
        transformed_filename = transform_filename(line)
        # 将转换后的数据写入文件
        file.write(transformed_filename + "\n")
        # 打印转换后的文件名（可选）
        print(transformed_filename)