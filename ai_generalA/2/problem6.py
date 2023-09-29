## 問5~7: ノイズ付きsin関数に関するプログラム
import numpy as np
import matplotlib.pyplot as plt

print("問6")
## サンプルデータD を入力
X = [0.349526784, 1.6974435, 5.384308891, 2.044150596,
     4.578814506, 3.241690807, 2.535931731, 2.210580888,
     3.397474351, 5.972933146, 5.114704101]

Y = [0.254020646, 0.790556868, -0.81239532, 1.012143475,
     -0.904558188, -0.167456361, 0.482547054, 0.878514378,
     -0.210093715, -0.128786937, -0.866501299]
## 勾配法を実装する(問6)
def gradient(data_x, data_y):
  ''' Ir: 学習率, epochs: 学習回数'''
  Ir = 0.000008
  epochs = 200000

  a = 0
  b = 0
  c = 0
  d = 0

  for epoch in range(epochs):
    grad_a = 0
    grad_b = 0
    grad_c = 0
    grad_d = 0
    loss = 0

    for i in range(len(data_x)):
      x = data_x[i]
      y = data_y[i]

      grad_a = grad_a - 2*x**3*(y - a*x**3 - b*x**2 - c*x - d)
      grad_b = grad_b - 2*x**2*(y - a*x**3 - b*x**2 - c*x - d)
      grad_c = grad_c - 2*x*(y - a*x**3 - b*x**2 - c*x - d)
      grad_d = grad_d - 2*(y - a*x**3 - b*x**2 - c*x - d)
      loss = loss + (y - a*x**3 - b*x**2 - c*x - d)**2

    a = a - Ir * grad_a
    b = b - Ir * grad_b
    c = c - Ir * grad_c
    d = d - Ir * grad_d

    ## ログが膨大であるため，1000 epochごとに出力
    if epoch % 1000 == 0:
      print("epoch:", epoch, "a:", a, "b:", b, "c:", c, "d:", d, "loss:", loss)

  return a, b, c, d

## np.array 変換
noize_x = np.array(X)
noize_y = np.array(Y)

a, b, c, d = gradient(X, Y)
usegrad_x = np.linspace(0, 6, 100)
usegrad_y = a*x**3 + b*x**2 + c*x + d


## sin関数を，0.1刻みで計算
x = np.linspace(0, 6, 100)
y = np.sin(x)

## 散布図・グラフを描画
plt.scatter(noize_x, noize_y, color="red")
plt.plot(usegrad_x, usegrad_y, color="blue")
plt.plot(x, y, color = "green")

## x軸の範囲は[0, 6]
plt.xlabel("x")
plt.ylabel("y")
plt.show()