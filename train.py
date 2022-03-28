"""
Implementation of AlexNet, from paper
"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.
See: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""

import torch
import torch.nn as nn  
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder 
from torchvision.transforms import transforms 
import os 
import argparse
from tqdm import tqdm

from model import AlexNet

# Create Dataloader object
def load_data(data_dir):
    image_transforms = transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor()])
    dataset = ImageFolder(data_dir, transform=image_transforms)
    train_len = int(len(dataset)*TRAIN_SPLIT)
    test_len = len(dataset)-train_len

    train_set, test_set = random_split(dataset=dataset, lengths=[train_len, test_len])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
    
    return train_loader, test_loader

# Load Checkpoint
def load_checkpoint(ckpt_path):
    checkpoint = os.listdir(ckpt_path)
    print(checkpoint)
    assert checkpoint.endswith(".pth"), "invalid checkpoint"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint)
    checkpoint_dict = torch.load(checkpoint_path)
    return checkpoint_dict

# Create a model based on existing model or new model
def get_model(use_pretrained=False, ckpt_path=None):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alexnet = AlexNet(NUM_CLASSES)
    optimizer = torch.optim.Adam(params=alexnet.parameters(), lr=LEARNING_RATE)
    if use_pretrained:
        ckpt = load_checkpoint(ckpt_path)
        alexnet.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        alexnet = alexnet.to(device)

    return alexnet, optimizer

# Training loop
def train(train_loader, model, loss_function, optimizer, save_ckpt=True):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    last_ckpt = None
    
    for epoch in range(NUM_EPOCHS):
        losses = []
        loop = tqdm(train_loader)
        for image, label in loop:

            image, label = image.to(device), label.to(device)
            model = model.to(device)
            prediction = model(image)

            loss = loss_function(prediction, label)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item())
            
        
        print(f"Loss at end of epoch {epoch+1} is {sum(losses)/len(losses)}")

        if save_ckpt:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), "loss": sum(losses)/len(losses)}, os.path.join(CHECKPOINT_DIR, f"alexnet_epoch_{epoch+1}.pth"))
            if last_ckpt is not None: os.remove(last_ckpt)
            last_ckpt = os.path.join(CHECKPOINT_DIR, f"alexnet_epoch_{epoch+1}.pth")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", help="No. of classes", default=2)
    parser.add_argument("--batch_size", help="Batch size", default=128)
    parser.add_argument("--epochs", help="No.of epochs", default=90)
    parser.add_argument("--train_split", help= "percentage of train set", default=0.8)
    parser.add_argument("--lr", help="Learning rate", default=0.0001)
    parser.add_argument("--ckpt_dir", help="Checkpoint dir", required=True)
    parser.add_argument("--data_dir", help="data dir", required=True)
    args = parser.parse_args()
    
    ## Arguments
    USE_PRETRAINED = True
    NUM_CLASSES = args.classes
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    TRAIN_SPLIT = args.train_split
    LEARNING_RATE = args.lr
    CHECKPOINT_DIR = args.ckpt_dir
    DATA_DIR = args.data_dir

    train_loader, test_loader = load_data(DATA_DIR)
    model, optimizer = get_model()
    loss_function = nn.CrossEntropyLoss()
    train(train_loader=train_loader, model=model, loss_function=loss_function, optimizer=optimizer)
