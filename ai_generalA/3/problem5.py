# 問5 pytorchを使った画像認識
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision as tv

## データセット作成(CIFAR10)
train_dataset = tv.datasets.CIFAR10(root="./", train=True,
                                  transform = tv.transforms.ToTensor(), download = True)
test_dataset = tv.datasets.CIFAR10(root="./", train=False,
                                  transform = tv.transforms.ToTensor(), download = True)

## データローダー作成
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

## グローバル変数の定義
MODELNAME = "cifar10_conv.model"
epochs = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CIFAR10_CONV(torch.nn.Module):
  def __init__(self):
    super(CIFAR10_CONV, self).__init__()
    ## モデル定義
    self.conv = torch.nn.Conv2d(3, 64, 3)  #out=64, kernel=3と設定
    # self.l1 = torch.nn.Linear(3072, 300)  # CIFER 10の次元
    self.l2 = torch.nn.Linear(64 * 30 * 30, 300)
    self.l3 = torch.nn.Linear(300, 10)

  ## 順伝搬
  def forward(self, x):
    x = x.reshape(-1, 3, 32, 32)
    h = F.relu(self.conv(x))  #convで扱えるようにreshape
    # h = F.relu(self.l1(x))
    h = h.reshape(100, -1)  #linearで扱えるようにreshape
    h = F.relu(self.l2(h))
    y = self.l3(h)
    return y

## 学習
def train():
  model = CIFAR10_CONV().to(DEVICE)  ## インスタンス，cudaに乗せる
  optimizer = torch.optim.Adam(model.parameters())

  for e in range(epochs):
    loss = 0
    for images, labels in train_loader:
      images = images.view(-1, 3*32*32).to(DEVICE) #view: 次元を指定して変形する
      optimizer.zero_grad()
      y = model(images).to(DEVICE) # cudaに乗せる
      labels = labels.to(DEVICE)  # cudaに乗せる

      batchloss = F.cross_entropy(y, labels)
      batchloss.backward()
      optimizer.step()
      loss = loss + batchloss.item()

    print("epoch: ", e, "loss: ", loss)
  torch.save(model.state_dict(), MODELNAME) ## モデルを保存

## テスト
pred_list = []
def test():
  correct = 0
  total = len(test_loader.dataset)

  ## モデルのロード
  model = CIFAR10_CONV().to(DEVICE)
  model.load_state_dict(torch.load(MODELNAME))

  model.eval() ## 評価モード

  for images, labels in test_loader:
    images = images.view(-1, 3*32*32).to(DEVICE) #view: 次元を指定して変形する
    y = model(images).to(DEVICE) # cudaに乗せる
    labels = labels.to(DEVICE)  # cudaに乗せる
    pred_labels = y.max(dim=1)[1] #最大出力
    correct = correct + (pred_labels == labels).sum() ## accuracyの分子
  print("correct:" , correct.item())
  print("total: ", total),
  print("accuracy:", (correct.item()/float(total)))

train()
test()