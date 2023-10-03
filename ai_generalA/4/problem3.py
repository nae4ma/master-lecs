# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MoSvAjnWAg-6j0zZGjJxXPAsxsFK8086
"""

import requests
import torch
import torch.nn.functional as F
import torchtext

## データのダウンロード
url = "https://github.com/odashi/small_parallel_enja/raw/master/"
train_en = [line.split() for line in requests.get(url+"train.en").text.splitlines()]
train_ja = [line.split() for line in requests.get(url+"train.ja").text.splitlines()]
test_en = [line.split() for line in requests.get(url+"test.en").text.splitlines()]
test_ja = [line.split() for line in requests.get(url+"test.ja").text.splitlines()]

for i in range(10):
  print(train_en[i])
  print(train_ja[i])
print("# of line", len(train_en), len(train_ja), len(train_en), len(train_ja))

## グローバル変数設定
#MODELNAME="tanaka-enja-rnn.model"
MODELNAME="tanaka-enja-lstm.model"
EPOCH = 20
BATCHSIZE = 128
LR = 0.001  ## Learning Rate は0.001に設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

## 語彙リストの作成
def make_vocab(train_data, min_freq):
  '''train_data:  学習データ，min_freq: トークン登録の基準'''
  vocab = {}
  for tokenlist in train_data:
    # トークンの出現回数カウント
    for token in tokenlist:
      if token not in vocab:
        vocab[token] = 0
      vocab[token] += 1

  # 語彙リストの予約
  vocablist = [('<unk>', 0), ('<pad>', 0), ('cls', 0), ('eos', 0)]
  vocabidx = {}

  for token, freq in vocab.items():
    # min_freq 以上のトークンだけ登録
    if freq >= min_freq:
      idx = len(vocablist)
      vocablist.append((token, freq))
      vocabidx[token] = idx
  vocabidx['<unk>'] = 0
  vocabidx['<pad>'] = 1
  vocabidx['<cls>'] = 2
  vocabidx['<eos>'] = 3

  return vocablist, vocabidx

vocablist_en, vocabidx_en = make_vocab(train_en, 3)  #min_freq = 3
vocablist_ja, vocabidx_ja = make_vocab(train_ja, 3)  #min_freq = 3

print("vocab")

## 未言語処理

def preprocess(data, vocabidx):
  '''vocabidx: 語彙リスト'''
  rr = []
  for tokenlist in data:
    tkl = ['<cls>']
    for token in tokenlist:
      #語彙リストにないものはunknown
      tkl.append(token if token in vocabidx else '<unk>')
    tkl.append('<eos>')
    rr.append(tkl)
  return rr

train_en_prep = preprocess(train_en, vocabidx_en)
train_ja_prep = preprocess(train_ja, vocabidx_ja)
test_en_prep = preprocess(test_en, vocabidx_en)

## データのzip化・ソート
train_data = list(zip(train_en_prep, train_ja_prep))
train_data.sort(key=lambda x: (len(x[0]), len(x[1])))
test_data = list(zip(test_en_prep, test_en, test_ja))

## ミニバッチの作成
def make_batch(data, batchsize):
  bb = []
  ben = []
  bja = []

  for en, ja in data:
    ben.append(en)
    bja.append(ja)

    # batchsizeと同じになると，bbに追加
    if len(ben) >= batchsize:
      bb.append((ben, bja))
      ben = []
      bja = []
  if len(ben) > 0:
    bb.append((ben, bja))
  return bb

train_data = make_batch(train_data, BATCHSIZE)

## パディング
def padding_batch(b):
  maxlen = max([len(x) for x in b])
  for tkl in b:
    for i in range(maxlen - len(tkl)):
      tkl.append('<pad>') #最大トークンと同じ長さになるまでpadding

def padding(bb):
  for ben, bja in bb:
    padding_batch(ben)
    padding_batch(bja)

padding(train_data)

## ID化
train_data =[([[vocabidx_en[token] for token in tokenlist] for tokenlist in ben],
              [[vocabidx_ja[token] for token in tokenlist] for tokenlist in bja]) for ben, bja in train_data]
test_data =[([vocabidx_en[token] for token in enprep], en, ja) for enprep, en, ja in test_data]

## エンコーダ・デコーダLSTM
class LSTMEncDec(torch.nn.Module):
  def __init__(self, vocablist_x, vocabidx_x, vocablist_y, vocabidx_y):
    super(LSTMEncDec, self).__init__()

    # encoder
    self.encemb = torch.nn.Embedding(len(vocablist_x), 256, padding_idx=vocabidx_x['<pad>'])
    self.encrcnn = torch.nn.LSTM(256, 516, 2, dropout=0.5)

    # decoder
    self.decemb = torch.nn.Embedding(len(vocablist_y), 256, padding_idx=vocabidx_y['<pad>'])
    self.decrcnn = torch.nn.LSTM(256, 516, 2, dropout=0.5)
    self.decout = torch.nn.Linear(516, len(vocablist_y))

    # dropout
    self.dropout = torch.nn.Dropout(0.5)

  def forward(self, x):
    x, y = x[0], x[1]
    # encoder
    e_x = self.encemb(x)
    e_x = self.dropout(e_x)

    out, (h,c) = self.encrcnn(e_x)

    # decoder
    e_y = self.decemb(y)
    e_y = self.dropout(e_y)
    n_y = e_y.size()[0]

    loss = torch.tensor(0., dtype=torch.float32).to(DEVICE)
    out, (h,c) = self.decrcnn(e_y[:-1], (h,c))

    for i in range(n_y - 1):
      loss += F.cross_entropy(self.decout(out[i]), y[i+1]) #損失を計算

    return loss

  # evaluate
  def evaluate(self, x, vocablist_y, vocabidx_y):
    #encoder
    e_x = self.encemb(x)
    e_x = self.dropout(e_x)
    n_x = e_x.size()[0]

    out, (h,c) = self.encrcnn(e_x)

    #decoder
    y = torch.tensor([vocabidx_y['<cls>']]).to(DEVICE)
    pred = []

    for i in range(50):
      e_y = self.decemb(y).unsqueeze(0)
      e_y = self.dropout(e_y)
      out, (h,c) = self.decrcnn(e_y, (h,c))

      pred_id = self.decout(out).squeeze().argmax()
      if pred_id == vocabidx_y['<eos>']:
        break
      pred_y = vocablist_y[pred_id][0]
      pred.append(pred_y)
      y[0] = pred_id
      # e_y = self.decemb(y)

    return pred

## 学習
def train():
  #model = RNNEncDec(vocablist_en, vocabidx_en, vocablist_ja, vocabidx_ja).to(DEVICE)
  model = LSTMEncDec(vocablist_en, vocabidx_en, vocablist_ja, vocabidx_ja).to(DEVICE)
  optimizer = torch.optim.Adam(model.parameters(), lr=LR) #Adamを使用

  for epoch in range(EPOCH):
    loss = 0
    for ben, bja in train_data:
      ben = torch.tensor(ben, dtype=torch.int64).transpose(0,1).to(DEVICE)
      bja = torch.tensor(bja, dtype=torch.int64).transpose(0,1).to(DEVICE)
      optimizer.zero_grad()
      batchloss = model((ben, bja))
      batchloss.backward()
      optimizer.step()
      loss = loss + batchloss.item()

    print("epoch:", epoch, "loss:", loss)
  torch.save(model.state_dict(), MODELNAME)


## テスト
def test():
  #model = RNNEncDec(vocablist_en, vocabidx_en, vocablist_ja, vocabidx_ja).to(DEVICE)
  model = LSTMEncDec(vocablist_en, vocabidx_en, vocablist_ja, vocabidx_ja).to(DEVICE)
  model.load_state_dict(torch.load(MODELNAME))
  model.eval()

  ref = []
  pred = []

  for enprep, en, ja in test_data:
    input = torch.tensor([enprep], dtype=torch.int64).transpose(0,1).to(DEVICE)
    p = model.evaluate(input, vocablist_ja, vocabidx_ja)

    if len(ref) < 20:
      print("INPUT;", en)
      print("REF;", ja)
      print("MT;", p)

    ref.append([ja])
    pred.append(p)

  bleu = torchtext.data.metrics.bleu_score(pred, ref)
  print(type(bleu))

  print("total:", len(test_data))
  print("bleu:", bleu)

#train()
test()