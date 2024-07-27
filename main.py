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
from data import make_dataset,make_preds,plot_preds,display_random_images,test_image
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
  train_loader, test_loader,test_data,train_data = make_dataset()
  # Train model
  train(model, train_loader,test_loader, device,args.epoch)

  if args.save:
    print(f"saving meowNet's state dict :3")
    torch.save(model.state_dict(),'meowNet.pth')

  if args.view_preds:
    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(test_data),k=9):
      test_samples.append(sample)
      test_labels.append(label)

    test_samples = torch.stack(test_samples).to(device)
    pred_probs = make_preds(model,[(test_samples, test_labels)],device)
    plot_preds(test_samples,pred_probs.squeeze().cpu(),test_labels,test_data.classes)

  if args.transform:
    display_random_images(train_data,train_data.classes,5,True,None)

  if args.test:
    test_image(model,args.test,device,train_data.classes)

if __name__ == '__main__':
  main()




