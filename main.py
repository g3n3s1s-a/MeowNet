#importing neccesary libs
import torch
from torch import nn
import torchvision
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import random
import matplotlib.pyplot as plt
import os
import random
import shutil
from torchvision.utils import make_grid
from tqdm import tqdm
from model import MeowNet, train
from data import make_dataset
import parse

def main():
  device = 'mps' if torch.backends.mps.is_available() else 'cpu'
  print(f"device used in this session: {device}")


  print("Making new model...")
  model = MeowNet()

  args = parse.args
  if args.load:
    print(f'loading in meownet state dict')
    model.load_state_dict(torch.load('meowNet.pth'))
    model.to(device)

  # get data
  train_loader, test_loader = make_dataset()

  # Train model
  train(model, train_loader,test_loader, device,args.epoch)
  if args.save:
    print(f"saving meowNet's state dict :3")
    torch.save(model.state_dict(),'meowNet.pth')


if __name__ == '__main__':
  main()




