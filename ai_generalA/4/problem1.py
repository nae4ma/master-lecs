!pip install torchdata
!pip install portalocker==2.7.0
import torch
import torch.nn.functional as F
import torchtext

train_iter, test_iter = torchtext.datasets.IMDB(split=('train', 'test'))
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

## グローバル変数
MODELNAME = "imdb-rnn.model"
EPOCH = 10
BATCHSIZE = 64
LR = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

## データの取り出し，トークン化，ソート
train_data = [(label, tokenizer(line)) for label, line in train_iter]
train_data.sort(key = lambda x: len(x[1]))
test_data = [(label, tokenizer(line)) for label, line in test_iter]
test_data.sort(key = lambda x: len(x[1]))

## 語彙リストの作成
def make_vocab(train_data, min_freq):
  '''train_data:  学習データ，min_freq: トークン登録の基準'''
  vocab = {}
  for label, tokenlist in train_data:
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

vocablist, vocabidx = make_vocab(train_data, 10)  #min_freq = 10
print(vocabidx)
## 未言語処理

def preprocess(data, vocabidx):
  '''vocabidx: 語彙リスト'''
  rr = []
  for label, tokenlist in data:
    tkl = ['<cls>']
    for token in tokenlist:
      #語彙リストにないものはunknown
      tkl.append(token if token in vocabidx else '<unk>')
    tkl.append('<eos>')
    rr.append((label, tkl))
  return rr

train_data = preprocess(train_data, vocabidx)
test_data = preprocess(test_data, vocabidx)

for i in range(10):
  print(train_data[i])

## ミニバッチの作成
def make_batch(data, batchsize):
  bb = []
  blabel = []
  btokenlist = []

  for label, tokenlist in data:
    blabel.append(label)
    btokenlist.append(tokenlist)
    # batchsizeと同じになると，bbに追加
    if len(blabel) >= batchsize:
      bb.append((btokenlist, blabel))
      blabel = []
      btokenlist = []
  if len(blabel) > 0:
    bb.append((btokenlist, blabel))
  return bb

train_data = make_batch(train_data, BATCHSIZE)
test_data = make_batch(test_data, BATCHSIZE)

## パディング
def padding(bb):
  for tokenlists, labels in bb:
    maxlen = max([len(x) for x in tokenlists])
    for tkl in tokenlists:
      for i in range(maxlen - len(tkl)):
        tkl.append('<pad>') #最大トークンと同じ長さになるまでpadding
  return bb

train_data = padding(train_data)
test_data = padding(test_data)

## ID化
def word2id(bb, vocabidx):
  rr = []
  for tokenlists, labels in bb:
    id_labels = [label - 1 for label in labels]
    id_tokenlists = []
    for tokenlist in tokenlists:
      id_tokenlists.append([vocabidx[token] for token in tokenlist])
    rr.append((id_tokenlists, id_labels))
  return rr

train_data = word2id(train_data, vocabidx)
test_data = word2id(test_data, vocabidx)

## クラスRNN
class MyRNN(torch.nn.Module):
  def __init__(self):
    super(MyRNN, self).__init__()
    vocabsize = len(vocablist)
    self.emb = torch.nn.Embedding(vocabsize, 300, padding_idx=vocabidx['<pad>'])  #埋め込み
    self.l1 = torch.nn.Linear(300, 300)
    self.l2 = torch.nn.Linear(300, 2)

  def forward(self, x):
    e = self.emb(x)
    h = torch.zeros(e[0].size(), dtype=torch.float32).to(DEVICE)
    for i in range(x.size()[0]):
      h = F.relu(e[i] + self.l1(h))
    return self.l2(h)

## 学習
def train():
  model = MyRNN().to(DEVICE)
  optimizer = torch.optim.Adam(model.parameters(), lr=LR) #Adamを使用

  for epoch in range(EPOCH):
    loss = 0
    for tokenlists, labels in train_data:
      tokenlists = torch.tensor(tokenlists, dtype=torch.int64).transpose(0,1).to(DEVICE)
      labels = torch.tensor(labels, dtype=torch.int64).to(DEVICE)
      optimizer.zero_grad()
      y = model(tokenlists)
      batchloss = F.cross_entropy(y, labels)
      batchloss.backward()
      optimizer.step()
      loss += batchloss.item()

    print("epoch:", epoch, "loss:", loss)
  torch.save(model.state_dict(), MODELNAME)


## テスト
def test():
  total = 0
  correct = 0

  model = MyRNN().to(DEVICE)
  model.load_state_dict(torch.load(MODELNAME))
  model.eval()  #評価モード

  for tokenlists, labels in test_data:
    total += len(labels)
    tokenlists = torch.tensor(tokenlists, dtype=torch.int64).transpose(0,1).to(DEVICE)
    labels = torch.tensor(labels, dtype=torch.int64).to(DEVICE)

    y = model(tokenlists)
    pred_labels = y.max(dim=1)[1]
    correct += (pred_labels == labels).sum()

  print("correct:", correct.item())
  print("total:", total)
  print("accuracy:", (correct.item() / float(total)))

train()
test()