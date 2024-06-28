import os
import random
import json


def listdir_nohidden(path):
    return [f for f in os.listdir(path) if not f.startswith('.')]


def generate_data(image_dir):
    data = {'train': [], 'val': [], 'test': []}
    index = 0
    # 获取所有类别（文件夹）
    categories = listdir_nohidden(image_dir)

    # 遍历每个类别
    for category in categories:
        category_path = os.path.join(image_dir, category)
        if os.path.isdir(category_path):
            # 获取类别下的所有图像文件
            images = listdir_nohidden(category_path)
            num_images = len(images)

            # 按照 8:1:1 的比例划分每个类别的数据
            num_train = int(0.4 * num_images)
            num_val = int(0.1 * num_images)
            num_test = num_images - num_train - num_val

            # 打乱图像文件列表
            random.shuffle(images)

            # 添加图像路径、索引和类别到数据列表中
            for i, image in enumerate(images):
                if i < num_train:
                    split = 'train'
                elif i < num_train + num_val:
                    split = 'val'
                else:
                    split = 'test'

                data[split].append([os.path.join(category, image), index, category])
        index += 1

    return data, categories


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
def save_categories(categories, filename):
    with open(filename, 'w') as f:
        for category in categories:
            f.write(f"{category}\n")
# 使用示例
image_directory = '/mnt/data/aid/AID'  # 替换为你的图像目录路径
output_json = 'aid_split.json'
categories_txt = 'categories_aid.txt'

# 生成数据列表
data_list, categories = generate_data(image_directory)

# 保存为 JSON 文件
save_json(data_list, output_json)
# 保存类别为 txt 文件
save_categories(categories, categories_txt)

print(f'JSON file "{output_json}" has been created successfully.')