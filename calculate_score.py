#this code is modified from 'https://github.com/gumusserv/CLIP-SalGan'
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from generator import *
from discriminator import *
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from score import *


from PIL import Image
import torchvision.transforms.functional as TF

choose_model = 'g2d2'

if choose_model == 'g1d1':
    generator =Generator()
    discriminator = Discriminator()
if choose_model == 'g1d2':
    generator =Generator()
    discriminator = Discriminator2()
if choose_model == 'g2d1':
    generator =Generator2()
    discriminator = Discriminator()
if choose_model == 'g2d2':
    generator =Generator2()
    discriminator = Discriminator2()
    

generator.load_state_dict(torch.load(choose_model+'\generator.pt', map_location = torch.device('cpu')))
discriminator.load_state_dict(torch.load(choose_model+'\discriminator.pt', map_location = torch.device('cpu')))

generator.eval()
discriminator.eval()



import json
import os



# 用于获取每个图片-文本对的图片路径, 文本描述
def get_Data(image_paths, target_paths):


    with open('text.json', 'r') as f:
        text_dic = json.load(f)

    text_descriptions = []

    for path in target_paths:
        path = path[path.rfind('/') + 1:]
        first_nonzero_index = None

        for i in range(len(path)):
            if path[i] != '0':
                first_nonzero_index = i
                break
        if first_nonzero_index != None:
            path = path[first_nonzero_index:]
        
        path = path[:path.find('.') ]
        
        text_descriptions.append(text_dic[path])

    return image_paths, target_paths, text_descriptions






# Define the transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

import torch
import clip
from PIL import Image

# 设置设备
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'

# 加载模型
model, preprocess = clip.load("ViT-B/32", device=device)

class SaliencyDatasetWithText(Dataset):
    def __init__(self, image_paths, target_paths, text_sequences, transform=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        
        
        self.text_sequences = []
        for i in range(len(image_paths)):
            # 处理图片
            # image = preprocess(Image.open(image_paths[i])).unsqueeze(0).to(device)
            
            # 处理文本
            text_tokens = clip.tokenize([text_sequences[i]]).to(device)

            with torch.no_grad():
                # 生成图片和文本的特征向量
                # image_features = model.encode_image(image)
                text_features = model.encode_text(text_tokens)

                # 打印或存储特征向量
                # print(text_features.cpu().numpy().shape)
                self.text_sequences.append(text_features)




        # self.text_sequences = text_to_embedding(text_sequences)
        # print(self.text_sequences.shape)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        target = Image.open(self.target_paths[idx]).convert('L')
        text = self.text_sequences[idx]

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        # 将文本序列转换为 PyTorch 张量
        text_tensor = torch.tensor(text, dtype=torch.long)

        return image, target, text_tensor
def create_dataloader(data, transform, batch_size=4, shuffle=True):
    image_paths, target_paths, text_descriptions = zip(*data)
    dataset = SaliencyDatasetWithText(list(image_paths), list(target_paths), list(text_descriptions), transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 添加批次维度
    return image




def resize_saliency_map(saliency_map, original_size):
    # 将PyTorch张量转换为PIL图像
    saliency_map_pil = TF.to_pil_image(saliency_map)
    # 调整大小
    resized_saliency_map = saliency_map_pil.resize(original_size, Image.BILINEAR)
    return resized_saliency_map

# 指定目标目录路径
image_directory_path = 'saliency/image'
target_directory_path = 'saliency/map'


with open('test_data_list_total.json', 'r') as f:
    test_data = json.load(f)

# image_paths_all = ['saliency/image_1800/000000000109_0.png', 'saliency/image_1800/000000000109_2.png', 'saliency/image_1800/000000000109_3.png']
# target_paths_all = ['saliency/map_1800/000000000109_0.png', 'saliency/map_1800/000000000109_2.png', 'saliency/map_1800/000000000109_3.png']
image_paths_all = []
target_paths_all = []
print(len(test_data))



for i in range(len(test_data)):
    image_paths_all.append(test_data[i][0])

    target_paths_all.append(test_data[i][1])
    




score_dic = dict()

score_dic['pure'] = {"AUC" : [], "sAUC" : [], "CC" : [], "NSS" : []}
score_dic['nonsal'] = {"AUC" : [], "sAUC" : [], "CC" : [], "NSS" : []}
score_dic['sal'] = {"AUC" : [], "sAUC" : [], "CC" : [], "NSS" : []}
score_dic['general'] = {"AUC" : [], "sAUC" : [], "CC" : [], "NSS" : []}
score_dic['total'] = {"AUC" : [], "sAUC" : [], "CC" : [], "NSS" : []}



for i in range(0, len(image_paths_all), 2):
    picture_list = []
    ground_truth = []

    for k in range(2):
    
        image_paths, target_paths, text_descriptions = get_Data([image_paths_all[i + k]], [target_paths_all[i + k]])
        # print(image_paths)
        # print(target_paths)
        if k == 0:
            ground_truth.append(target_paths[0])
            picture_list.append(target_paths[0])
        ground_truth.append(target_paths[0])
        # print(text_descriptions)


        val_data = list(zip(image_paths, target_paths, text_descriptions))

        val_loader = create_dataloader(val_data, transform)

        criterion = nn.BCELoss()
        with torch.no_grad():
            val_loss = 0.0
            val_loss2 = 0.0
            for m, (images, targets, texts_embeddings) in enumerate(val_loader):
                # print(image_paths[m])
                # 从文件获取原始图像尺寸
                original_image = Image.open(image_paths[m])
                original_size = original_image.size
                # 只计算生成器的损失
                fake_targets = generator(images, texts_embeddings)
                outputs = discriminator(fake_targets,texts_embeddings)
                val_loss += criterion(outputs, torch.ones(images.size(0), 1)).item()
                picture = fake_targets.squeeze(0)

                AUC_score = AUC(fake_targets, targets)
                sAUC_score = sAUC(fake_targets, targets)
                CC_score = CC(fake_targets, targets)
                NSS_score = NSS(fake_targets, targets)

                # print("AUC Score: {}".format(AUC_score))
                # print("sAUC Score: {}".format(sAUC_score))
                # print("CC Score: {}".format(CC_score))
                # print("NSS Score: {}".format(NSS_score))

                if "_2.png" in image_paths[m]:
                    # print(2)
                    score_dic['nonsal']['AUC'].append(AUC_score)
                    score_dic['nonsal']['sAUC'].append(sAUC_score)
                    score_dic['nonsal']['CC'].append(CC_score)
                    score_dic['nonsal']['NSS'].append(NSS_score)
                elif "_3.png" in image_paths[m]:
                    # print(3)
                    score_dic['sal']['AUC'].append(AUC_score)
                    score_dic['sal']['sAUC'].append(sAUC_score)
                    score_dic['sal']['CC'].append(CC_score)
                    score_dic['sal']['NSS'].append(NSS_score)
                elif "_0.png" in image_paths[m]:
                    # print(0)
                    score_dic['pure']['AUC'].append(AUC_score)
                    score_dic['pure']['sAUC'].append(sAUC_score)
                    score_dic['pure']['CC'].append(CC_score)
                    score_dic['pure']['NSS'].append(NSS_score)
                elif "_1.png" in image_paths[m]:
                    # print(1)
                    score_dic['general']['AUC'].append(AUC_score)
                    score_dic['general']['sAUC'].append(sAUC_score)
                    score_dic['general']['CC'].append(CC_score)
                    score_dic['general']['NSS'].append(NSS_score)
                score_dic['total']['AUC'].append(AUC_score)
                score_dic['total']['sAUC'].append(sAUC_score)
                score_dic['total']['CC'].append(CC_score)
                score_dic['total']['NSS'].append(NSS_score)

                # print()


                picture = resize_saliency_map(picture, original_size)

                


                picture_list.append(picture)
            

                # print(val_loss)
                
print('total')
print(sum(score_dic['total']['AUC']) / len(score_dic['total']['AUC']))
print(sum(score_dic['total']['sAUC']) / len(score_dic['total']['sAUC']))
print(sum(score_dic['total']['CC']) / len(score_dic['total']['CC']))
print(sum(score_dic['total']['NSS']) / len(score_dic['total']['NSS']))

print('pure')

# print(len(score_dic['pure']['AUC']))
print(sum(score_dic['pure']['AUC']) / len(score_dic['pure']['AUC']))
print(sum(score_dic['pure']['sAUC']) / len(score_dic['pure']['sAUC']))
print(sum(score_dic['pure']['CC']) / len(score_dic['pure']['CC']))
print(sum(score_dic['pure']['NSS']) / len(score_dic['pure']['NSS']))
print('sal')
print(sum(score_dic['sal']['AUC']) / len(score_dic['sal']['AUC']))
print(sum(score_dic['sal']['sAUC']) / len(score_dic['sal']['sAUC']))
print(sum(score_dic['sal']['CC']) / len(score_dic['sal']['CC']))
print(sum(score_dic['sal']['NSS']) / len(score_dic['sal']['NSS']))

print('nonsal')
print(sum(score_dic['nonsal']['AUC']) / len(score_dic['nonsal']['AUC']))
print(sum(score_dic['nonsal']['sAUC']) / len(score_dic['nonsal']['sAUC']))
print(sum(score_dic['nonsal']['CC']) / len(score_dic['nonsal']['CC']))
print(sum(score_dic['nonsal']['NSS']) / len(score_dic['nonsal']['NSS']))



print('general')
print(sum(score_dic['general']['AUC']) / len(score_dic['general']['AUC']))
print(sum(score_dic['general']['sAUC']) / len(score_dic['general']['sAUC']))
print(sum(score_dic['general']['CC']) / len(score_dic['general']['CC']))
print(sum(score_dic['general']['NSS']) / len(score_dic['general']['NSS']))

