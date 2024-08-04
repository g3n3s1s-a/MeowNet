import torch
from torch import nn
import tqdm
from tqdm import tqdm
import torch.nn.functional as F
class MeowNet(nn.Module):
    def __init__(self):
        super(MeowNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 1 * 1, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 12)  # Adjusted for 12 breeds

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Maintain batch size, flattening the rest
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)  # Output should be 12
        return x

def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
  acc = (correct / len(y_pred)) * 100
  return acc

def train_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              accuracy_fn,
              device):
  ''' performs a training with model trying to learn on data_loader'''

  train_loss,train_acc = 0,0
  model.train()
  # add a loop to loop through the training btaches
  for batch, (X,y) in enumerate(data_loader):

    X,y = X.to(device),y.to(device)
    #1. forward pass
    y_pred = model(X)

    #2. calculate loss (per batch)
    loss = loss_fn(y_pred,y)
    train_loss += loss # accumlate train loss
    train_acc += accuracy_fn(y_true=y,
                            y_pred=y_pred.argmax(dim=1))

    #3. optimizer zero grad
    optimizer.zero_grad()

    #4 loss backward
    loss.backward()

    #5. optimizer step
    optimizer.step()


  #divide total train loss by length of dataloader
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  print(f'Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}\n')

def test_step(
  model: torch.nn.Module,
  data_loader: torch.utils.data.DataLoader,
  loss_fn: torch.nn.Module,
  accuracy_fn,
  device
  ):
  ''' performs a testing loop step on a model going over data_loader'''

  test_loss, test_acc = 0,0
  model.eval()

  with torch.no_grad():
    for X,y in data_loader:
      X,y = X.to(device), y.to(device)

      test_pred = model(X)

      test_loss += loss_fn(test_pred,y)
      test_acc += accuracy_fn(y_true=y,
                              y_pred=test_pred.argmax(dim=1))

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f'test loss: {test_loss:.5f} | test acc: {test_acc:.2f}\n')

def train(model, train_dataloader,test_dataloader, device,epoch):
  loss_fn = nn.CrossEntropyLoss() # measures how wrong the model is
  optimizer = torch.optim.Adam(params=model.parameters(),lr= 0.00001)
  EPOCHS = epoch
  model = model.to(device)
  if epoch > 0:
    for epoch in tqdm(range(EPOCHS)):
      print(f'Epoch: {epoch}\n--------')
      train_step(model,train_dataloader,loss_fn,optimizer,accuracy_fn, device)
      test_step(model,test_dataloader,loss_fn,accuracy_fn,device)

