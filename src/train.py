## define & run training loops

import torch.utils
import torch.utils.data
import yaml
import torch
import torchvision
import torch.nn as nn 
import torch.optim as optim
from torchvision import transforms

from .models import TinyCNN

## load config from yaml

with open("configs/mnist.yaml") as file:
    config = yaml.safe_load(file)

## dataset & loader
## convert to 3 channels and 224 x 224 
transform = transforms.Compose([
                                transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root=config["data"]["root"], train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)

## TODO: replace with tiny CNN
model = TinyCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print('moved model to ' + ("cuda" if torch.cuda.is_available() else "cpu"))

## Optimizer
if config["training"]["optimizer"] == "adam":
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])
elif config["training"]["optimizer"] == "sgd":
    optimizer = optim.SGD(params=model.parameters(), lr=config["training"]["lr"])

loss_func = nn.CrossEntropyLoss()

## training loop
print("beginning train loop")
for epoch in range(config["training"]["epochs"]):
    epoch_loss = 0
    for example, label in train_loader:
        optimizer.zero_grad()
        output = model(example)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch}: Loss={epoch_loss/len(train_loader):.4f}")

