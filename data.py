from pathlib import Path
import os
import torchvision
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms
from typing import Tuple, Dict, List
from torch.utils.data import DataLoader
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import random

train_transforms = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.05, contrast=0.01, saturation=0.05, hue=0.05),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor()
])

# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):

    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:

        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*.jpg")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes = class_names(targ_dir)
        self.classes_to_idx = {name:i for i,name in enumerate(self.classes)}

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img_path = self.paths[index]
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        # Extract class name from filename
        if '-' in img_path.name:
          class_name = img_path.name.split('-')[0]
          if class_name == 'Russian':
            class_name = 'Russian_Blue'
          elif class_name == 'British':
            class_name = 'British_Shorthair'
          elif class_name == 'Egyptian':
            class_name = 'Egyptian_Mau'
          elif class_name == 'Maine':
            class_name = 'Maine_Coon'
        else:
          class_name = img_path.name.split('_')[0]
          if class_name == 'Russian':
            class_name = 'Russian_Blue'
          elif class_name == 'British':
            class_name = 'British_Shorthair'
          elif class_name == 'Egyptian':
            class_name = 'Egyptian_Mau'
          elif class_name == 'Maine':
            class_name = 'Maine_Coon'
        class_idx = self.classes_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)

def walk_through_dir(dir_path):
  ''' walks through dir path returning its content'''
  for root,dirnames,filenames in os.walk(dir_path):
    print(f'There are {len(dirnames)} directories and {len(filenames)} images in {root}')

def class_names(dir_path):
  classes = set()
  for filename in os.listdir(dir_path):
    if filename.endswith(".jpg"):
      if '-' in filename:
        label = filename.split('-')[0]
        if label == 'Russian':
          label = 'Russian_Blue'
        elif label == 'British':
          label = 'British_Shorthair'
        elif label == 'Egyptian':
          label = 'Egyptian_Mau'
        elif label == 'Maine':
          label = 'Maine_Coon'

      else:
        label = filename.split('_')[0]
        if label == 'Russian':
          label = 'Russian_Blue'
        elif label == 'British':
          label = 'British_Shorthair'
        elif label == 'Egyptian':
          label = 'Egyptian_Mau'
        elif label == 'Maine':
          label = 'Maine_Coon'
      classes.add(label)

  return sorted(list(classes))


def make_dataset():
  dataset_path = Path('/Users/genesisargueta/Desktop/2024 summer projects/MeowNet/dataset')
  train_dir = dataset_path / 'train'
  test_dir = dataset_path / 'test'
  class_name= class_names(test_dir)
  class_to_idx = {name:i for i,name in enumerate(class_name)}
  train_data = ImageFolderCustom(targ_dir=train_dir, transform=train_transforms)
  test_data = ImageFolderCustom(targ_dir=test_dir, transform=test_transforms)
  #display_random_images(train_data, n=5, classes=class_name, seed=None)

  BATCH_SIZE=32
  NUM_WORKERS = 2
  train_dataloader = DataLoader(dataset=train_data, # use custom created train Dataset
                                      batch_size=BATCH_SIZE, # how many samples per batch?
                                      num_workers=NUM_WORKERS, # how many subprocesses to use for data loading? (higher = more)
                                      shuffle=True) # shuffle the data?

  test_dataloader = DataLoader(dataset=test_data, # use custom created test Dataset
                                      batch_size=BATCH_SIZE,
                                      num_workers=NUM_WORKERS,
                                      shuffle=False) # don't usually need to shuffle testing data

  return train_dataloader, test_dataloader,test_data,train_data

def make_preds(model: torch.nn.Module,
                     data: list,
                     device):
  pred_probs = []
  model.eval()
  with torch.inference_mode():
    for X,y_true in data:
      #prepare the same (add a batch dim and pass to target device")
      X,y_true = X.to(device),y_true
      # Forward pass
      pred_logit = model(X)

      #get prediction prob
      pred_prob = pred_logit.argmax(dim=1)

      #get pred prob off gpu for further calculations
      pred_probs.append(pred_prob.cpu())

  # stack the pred_probs to turn list into a tensor
  return torch.stack(pred_probs)


def plot_preds(test_samples, pred_classes, true_labels, class_names):
    plt.figure(figsize=(12, 12))
    nrows = 3
    ncols = 3

    for i, sample in enumerate(test_samples):
        # Create subplot
        sample=sample.cpu()
        plt.subplot(nrows, ncols, i + 1)

        # Plot target image
        plt.imshow(sample.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C) for plotting

        # Find prediction in text form
        pred_label = class_names[pred_classes[i]]

        # Get the true label in text form
        truth_label = class_names[true_labels[i]]

        # Create a title for the plot
        title_text = f'Pred: {pred_label} | Truth: {truth_label}'

        # Check for equality between pred and truth and change color of title text
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, color='g')
        else:
            plt.title(title_text, fontsize=10, color='r')

        plt.axis('off')  # Hide axes
    plt.show()

def display_random_images(dataset: torch.utils.data.dataset.Dataset,classes: List[str] = None,n: int = 10,display_shape: bool = True,seed: int = None):

    # 2. Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")

    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)
    # 5. Setup plot
    plt.figure(figsize=(16, 8))
    plt.subplots_adjust(wspace=1.2, hspace=0.6)

    # 6. Loop through samples and display random samples
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
    plt.show()

def test_image(model,path,device,classes):
 #1. find the image and turn image to tensor
  img = Image.open(path)
  #2 transform img
  img = test_transforms(img)
  #3. Test img
  pred_probs = make_preds(model,[(img.unsqueeze(dim=0),0)],device)

  #4. Print image and return what the model guessed it was
  plt.subplot(3,3,2)
  img = img.cpu()
  pred_class = pred_probs.squeeze().cpu()
  plt.imshow(img.permute(1,2,0))
  title_text = f' Prediction: {classes[pred_class]}'
  plt.title(title_text,fontsize=10,color='mediumvioletred')
  plt.axis('off')
  plt.show()





