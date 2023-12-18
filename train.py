import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from Data_Utils import *
from get_data import *
from generator import *
from discriminator import *
from train import *
import json

def train_model(train_loader, val_loader, generator, discriminator, criterion, optimizer_G, optimizer_D, device, num_epochs=50):
    record_dic = dict()
    for epoch in range(num_epochs):
        epoch_dic = dict()
        generator.train()
        discriminator.train()
        
        running_loss_G = 0.0
        running_loss_D = 0.0

        for i, (images, targets, texts_embeddings) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            texts_embeddings = texts_embeddings.to(device)
            # Train Discriminator 
            optimizer_D.zero_grad()

            # Real samples
            real_labels = torch.ones(images.size(0), 1).to(device)    
            outputs = discriminator(targets,texts_embeddings)
            d_loss_real = criterion(outputs, real_labels)

            # Fake samples
            fake_targets = generator(images, texts_embeddings)
            fake_labels = torch.zeros(images.size(0), 1).to(device)
            outputs = discriminator(fake_targets.detach(),texts_embeddings)
            d_loss_fake = criterion(outputs, fake_labels)

            # total loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            outputs = discriminator(fake_targets,texts_embeddings)
            g_loss = criterion(outputs, real_labels)

            g_loss.backward()
            optimizer_G.step()

            running_loss_G += g_loss.item()
            running_loss_D += d_loss.item()

            
            if (i + 1) % 10 == 0:  
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Generator Loss: {running_loss_G / (i + 1)}, Discriminator Loss: {running_loss_D / (i + 1)}')
            step_dic = dict()
            step_dic['G LOSS'] = running_loss_G / (i + 1)
            step_dic['D LOSS'] = running_loss_D / (i + 1)
            epoch_dic[f"Step [{i + 1}/{len(train_loader)}]"] = step_dic

        
        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            val_loss = 0.0
            for images, targets, texts_embeddings in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                texts_embeddings = texts_embeddings.to(device)
                fake_targets = generator(images, texts_embeddings)
                outputs = discriminator(fake_targets,texts_embeddings)
                val_loss += criterion(outputs, torch.ones(images.size(0), 1).to(device)).item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: G - {g_loss.item()}, D - {d_loss.item()}, Val Loss: {val_loss / len(val_loader)}')


        step_dic = dict()
        step_dic["Train G Loss"] = g_loss.item()
        step_dic["Train D Loss"] = d_loss.item()
        step_dic["Val Loss"] = val_loss / len(val_loader)
        epoch_dic["Final"] = step_dic
        record_dic[epoch] = epoch_dic
    with open('g2d2/loss.json', 'w') as f:
        json.dump(record_dic, f)

if __name__ == '__main__':

    image_directory_path = 'saliency/image'
    target_directory_path = 'saliency/map'

    image_paths, target_paths, text_descriptions = get_Data(image_directory_path, target_directory_path)

    train_data, val_data, test_data = split_dataset(image_paths, target_paths, text_descriptions)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    batch_size = 16
    train_loader = create_dataloader(train_data, transform, batch_size=batch_size)
    val_loader = create_dataloader(val_data, transform, batch_size=batch_size)
    test_loader = create_dataloader(test_data, transform, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #generator = Generator().to(device)
    generator =Generator2().to(device)
    discriminator = Discriminator2().to(device)

    criterion = nn.BCELoss()
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.3)
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.3)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.00002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999))
    num_epochs = 50

    train_model(train_loader, val_loader, generator, discriminator, criterion, optimizer_G, optimizer_D, device, num_epochs)

    torch.save(generator.state_dict(), 'g2d2/generator.pt')
    torch.save(discriminator.state_dict(), 'g2d2/discriminator.pt')






