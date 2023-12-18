from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import torch
import clip
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)



class SaliencyDatasetWithText(Dataset):
    def __init__(self, image_paths, target_paths, text_sequences, transform=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.text_sequences = []
        for i in range(len(image_paths)):
            text_tokens = clip.tokenize([text_sequences[i]]).to(device)

            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                self.text_sequences.append(text_features)

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

        text_tensor = torch.tensor(text, dtype=torch.long)

        return image, target, text_tensor


    
def split_dataset(image_paths, target_paths, text_descriptions, train_ratio=0.7, val_ratio=0.15):
    combined = list(zip(image_paths, target_paths, text_descriptions))
    random.shuffle(combined)

    total_images = len(combined)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)

    train_data = combined[:train_size]
    val_data = combined[train_size:train_size + val_size]
    test_data = combined[train_size + val_size:]

    with open('test_data_list_total.json', 'w') as json_file:
        json.dump(test_data, json_file)
    

    return train_data, val_data, test_data


def create_dataloader(data, transform, batch_size = 32, shuffle=True):
    image_paths, target_paths, text_descriptions = zip(*data)
    dataset = SaliencyDatasetWithText(list(image_paths), list(target_paths), list(text_descriptions), transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


