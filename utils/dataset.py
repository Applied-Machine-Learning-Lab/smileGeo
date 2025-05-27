# -*- coding: utf-8 -*-
# @Time : 2024/3/15 17:43
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: GLM
# @File : dataset.py
# @Software: PyCharm
import os
import random
import torch
import pickle as pkl

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import warnings

warnings.filterwarnings("ignore")


class GeoDataset(Dataset):
    def __init__(self,
                 root="../data"
                 ):
        self.root = root
        df = pd.read_csv(self.root + '/location.csv', sep=';')
        self.count = 0
        self.imgs = []
        self.cities = {}
        self.countries = {}
        self.labels = {}
        self.names = {}
        self.makes = {}
        for index, row in df.iterrows():
            id = row['index']
            city = str(row['city'])
            country = str(row['country'])
            label = row['hierarchical_label']
            name = row['category'].replace("_", " ")
            madeby = row['natural_or_human_made']
            for jpgfile in row['images'].split(" "):
                self.imgs.append(jpgfile)
                self.insert_tree(self.cities, jpgfile, city)
                self.insert_tree(self.countries, jpgfile, country)
                self.insert_tree(self.labels, jpgfile, label)
                self.insert_tree(self.names, jpgfile, name)
                self.insert_tree(self.makes, jpgfile, madeby)
                self.count += 1
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),  # 调整图片大小为512x512
            transforms.ToTensor()  # 转换为Tensor
        ])

    def insert_tree(self, tree, key, value):
        assert type(key) == str, 'insert_tree: Error Type!'
        node = tree
        for letter in key:
            if letter not in node:
                node[letter] = {}
            node = node[letter]
        node['value'] = value
        return tree

    def get_value(self, tree, key):
        node = tree
        for letter in key:
            if letter not in node:
                return -1
            node = node[letter]
        if 'value' not in node:
            return -1
        else:
            return node['value']

    def __getitem__(self, index):
        img = self.imgs[index]
        city = self.get_value(self.cities, img)
        country = self.get_value(self.countries, img)
        label = self.get_value(self.labels, img)
        name = self.get_value(self.names, img)
        madeby = self.get_value(self.makes, img)
        img_name = img + '.jpg'
        # img_path = self.root + '/pic/' + img_name
        # image = Image.open(img_path)
        # image_tensor = self.transform(image)
        return img_name, city, country, label, name, madeby

    def __len__(self):
        return self.count


if __name__ == '__main__':
    geodata = GeoDataset()
    dataloader = DataLoader(geodata, batch_size=2, shuffle=True)

    for i, batch in enumerate(dataloader):
        print(i)
        print(batch)
        break