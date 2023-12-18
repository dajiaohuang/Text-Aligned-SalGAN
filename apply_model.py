import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from generator import *
from discriminator import *
import clip
import torchvision.transforms.functional as TF




def resize_saliency_map(saliency_map, original_size):
    # 将PyTorch张量转换为PIL图像
    saliency_map_pil = TF.to_pil_image(saliency_map)
    # 调整大小
    resized_saliency_map = saliency_map_pil.resize(original_size, Image.BILINEAR)
    
    return resized_saliency_map

class SaliencyDatasetWithText(Dataset):
    def __init__(self, image_paths, text_sequences, transform=None):
        self.image_paths = image_paths
        self.text_sequences = []
        for i in range(len(image_paths)):
            # 处理文本
            text_tokens = clip.tokenize([text_sequences[i]]).to(device)

            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                self.text_sequences.append(text_features)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        text = self.text_sequences[idx]

        if self.transform:
            image = self.transform(image)
            

        # 将文本序列转换为 PyTorch 张量
        text_tensor = torch.tensor(text, dtype=torch.long)

        return image, text_tensor


def create_dataloader(data, transform, batch_size = 1, shuffle=True):
    image_paths, text_descriptions = zip(*data)
    dataset = SaliencyDatasetWithText(list(image_paths), list(text_descriptions), transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)






if __name__ == '__main__':
    # 创建参数
    parser = argparse.ArgumentParser(
    description='',
    formatter_class=argparse.RawTextHelpFormatter
)
    parser.add_argument("--model", help="Choose the model to use. Options are 'g1d1', 'g1d2', 'g2d1', 'g2d2'.",\
                         choices=['g1d1', 'g1d2', 'g2d1', 'g2d2'], default='g1d2')
    parser.add_argument("--text", help="Choose the text type. Options are 'sal', 'nonsal', 'general','pure'.", \
                        choices=['sal', 'nonsal', 'general','pure'], default='pure')
    args = parser.parse_args()

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])


    # 设置设备
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device='cpu'

    # 加载模型
    model, preprocess = clip.load("ViT-B/32", device=device)


    #generator = Generator()
    if args.model == 'g1d1':
        generator =Generator()
        discriminator = Discriminator()
    if args.model == 'g1d2':
        generator =Generator()
        discriminator = Discriminator2()
    if args.model == 'g2d1':
        generator =Generator2()
        discriminator = Discriminator()
    if args.model == 'g2d2':
        generator =Generator2()
        discriminator = Discriminator2()

    
    # generator.load_state_dict(torch.load('generator.pt', map_location=torch.device('cpu')))
    # discriminator.load_state_dict(torch.load('discriminator.pt', map_location=torch.device('cpu')))
    
    generator.load_state_dict(torch.load(args.model +'/generator.pt', map_location=torch.device('cpu')))
    discriminator.load_state_dict(torch.load(args.model +'/discriminator.pt', map_location=torch.device('cpu')))

    # 模型进入评估模式
    generator.eval()
    discriminator.eval()

    # 待测验图片路径以及文本描述
    image_paths = ['saliency/image/000000001306_0.png']
    text_options = {
        "sal": "A table with chairs, a counter with stools and a big window",
        "nonsal": "A television was mounted on the kitchen wall.",
        "general": "A room with a TV, a table, a bar and floor-to-ceiling Windows",
        'pure': ''
    }
    # image_paths = ['saliency/image/000000026503_0.png']
    # text_options = {
    #     "sal": "Two dogs and a cat are lying on the bed",
    #     "nonsal": "There are white lamps on the bedside table",
    #     "general": "Three animals are lying on the bed, next to the white lamp"
    # }
    selected_text = text_options[args.text]
    texts_descriptions = [selected_text]


    # 创建输入数据
    val_data = list(zip(image_paths, texts_descriptions))
    
    val_loader = create_dataloader(val_data, transform)

    # 预测并生成图像
    with torch.no_grad():
        for m, (images, texts_embeddings) in enumerate(val_loader):
            # 从文件获取原始图像尺寸
            original_image = Image.open(image_paths[m])
            original_size = original_image.size
            # 只计算生成器的损失
            fake_targets = generator(images, texts_embeddings)
            picture = fake_targets.squeeze(0)

            picture = resize_saliency_map(picture, original_size)
            
            
            
    picture.save(args.model+'/'+args.text+'.png')
    # 创建一个两行三列的网格
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1, figure=fig)

    

    # 原图片
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_image)
    ax1.axis('off')

    # 预测显著图
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(picture, cmap='gray')
    ax2.axis('off')

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.1, hspace=0.0)

    # 显示图片
    plt.show()

                
    

    
    




