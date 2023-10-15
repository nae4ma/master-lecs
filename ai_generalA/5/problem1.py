# 問1 CIFAR10 精度向上に向けて
"""
参考:
[1]Karen S. and Andrew Z., 
VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION, 
ICLR 2015

[2]pytorch-cifar10, https://github.com/Ksuryateja/pytorch-cifar10
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision import models

from torchvision.models import VGG16_Weights

## データセット作成(CIFAR10)
train_dataset = tv.datasets.CIFAR10(root="./", train=True,
                                  transform = tv.transforms.ToTensor(), download = True)
test_dataset = tv.datasets.CIFAR10(root="./", train=False,
                                  transform = tv.transforms.ToTensor(), download = True)

## データローダー作成
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

## グローバル変数の定義
MODELNAME = "mycnn.model"
epochs = 20 #レポート課題3(問5)からの変更箇所
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## VGG 16の実装 (レポート課題3(問5)からの変更箇所)
# モデル構造
cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.fc = nn.Linear(512,512)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc(out)
        out = self.classifier(out)
        return out

    # 構造のcfgを元に，モデルを構成する
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

## 学習
def train():
  net = VGG(vgg_name="VGG16").to(DEVICE)  #レポート課題3(問5)からの変更箇所
  net.train()
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001) #レポート課題3(問5)からの変更箇所

  for e in range(epochs):
    loss = 0
    for images, labels in train_loader:
      images = images.to(DEVICE) #view: 次元を指定して変形する
      optimizer.zero_grad()
      y = net(images).to(DEVICE) # cudaに乗せる
      labels = labels.to(DEVICE)  # cudaに乗せる

      batchloss = F.cross_entropy(y, labels)
      batchloss.backward()
      optimizer.step()
      loss = loss + batchloss.item()

    print("epoch: ", e, "loss: ", loss)
  torch.save(net.state_dict(), MODELNAME) ## モデルを保存

## テスト
pred_list = []
def test():
  correct = 0
  total = len(test_loader.dataset)

  ## モデルのロード
  net = VGG(vgg_name="VGG16").to(DEVICE) #レポート課題3(問5)からの変更箇所
  net.to(DEVICE)
  net.load_state_dict(torch.load(MODELNAME))

  net.eval() # 評価モード

  for images, labels in test_loader:
    images = images.to(DEVICE) #view: 次元を指定して変形する
    y = net(images).to(DEVICE) # cudaに乗せる
    labels = labels.to(DEVICE)  # cudaに乗せる
    pred_labels = y.max(dim=1)[1] #最大出力
    correct = correct + (pred_labels == labels).sum() ## accuracyの分子
  print("correct:" , correct.item())
  print("total: ", total),
  print("accuracy:", (correct.item()/float(total)))

train()
test()