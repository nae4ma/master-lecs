## 問1: MNISTを用いた画像認識
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision as tv

## データセット作成
train_dataset = tv.datasets.MNIST(root="./", train=True,
                                  transform = tv.transforms.ToTensor(), download = True)
test_dataset = tv.datasets.MNIST(root="./", train=False,
                                  transform = tv.transforms.ToTensor(), download = True)

## データローダー作成
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

## グローバル変数の定義
epochs = 10
hidden = 300  ##中間層


## ネットワーク，パラメータ設定
linear1 = torch.nn.Linear(784, hidden) #784 -> 300
linear2 = torch.nn.Linear(hidden, 10) #300 -> 10
params = list(linear1.parameters()) + list(linear2.parameters())
optimizer = torch.optim.Adam(params)

def model(x):
  h = F.relu(linear1(x))
  y = linear2(h)
  return y

## 学習
for e in range(epochs):
  loss = 0
  for images, labels in train_loader:
    images = images.view(-1, 28*28) #view: 次元を指定して変形する
    optimizer.zero_grad()
    y = model(images)
    labels = labels
    batchloss = F.cross_entropy(y, labels)
    batchloss.backward()
    optimizer.step()
    loss = loss + batchloss.item()
  print("epoch: ", e, "loss: ", loss)

## テスト
correct = 0
total = len(test_loader.dataset)

for images, labels in test_loader:
  images = images = images.view(-1, 28*28) #view: 次元を指定して変形する
  y = model(images)
  pred_labels = y.max(dim=1)[1] #最大出力
  correct = correct + (pred_labels == labels).sum() ## accuracyの分子

print("correct:" , correct.item())
print("total: ", total),
print("accuracy:", correct.item()/total)