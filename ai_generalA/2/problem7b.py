## 問5~7: ノイズ付きsin関数に関するプログラム
import numpy as np
import matplotlib.pyplot as plt

print("問7b")
## サンプルデータD を入力
X = [0.349526784, 1.6974435, 5.384308891, 2.044150596,
     4.578814506, 3.241690807, 2.535931731, 2.210580888,
     3.397474351, 5.972933146, 5.114704101]

Y = [0.254020646, 0.790556868, -0.81239532, 1.012143475,
     -0.904558188, -0.167456361, 0.482547054, 0.878514378,
     -0.210093715, -0.128786937, -0.866501299]

## 解析解を求める(問7b)
def lsm_dim9(X, Y):
  X9 = []
  for x in X:
    X9 = X9 + [[1, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9]]
  X9 = np.array(X9)
  Y9 = np.array([Y]).T

  ## 正規方程式を解く
  Z1 = np.matmul(X9.T, X9)
  Z2 = np.linalg.inv(Z1)
  Z3 = np.matmul(Z2, X9.T)
  w = np.matmul(Z3, Y9)

  # パラメータを返す
  return w[9][0], w[8][0], w[7][0], w[6][0], w[5][0], w[4][0], w[3][0], w[2][0], w[1][0], w[0][0]

## np.array 変換
noize_x = np.array(X)
noize_y = np.array(Y)

w9, w8, w7, w6, w5, w4, w3, w2, w1, w0 = lsm_dim9(X, Y)
lsm_x = np.linspace(0, 6, 100)
lsm_y = w9*x**9 + w8*x**8 + w7*x**7 + w6*x**6 + w5*x**5 + w4*x**4 + w3*x**3 + w2*x**2 + w1*x + w0

## sin関数を，0.1刻みで計算
x = np.linspace(0, 6, 100)
y = np.sin(x)

## 散布図・グラフを描画
plt.scatter(noize_x, noize_y, color="red")
plt.plot(lsm_x, lsm_y, color="blue")
plt.plot(x, y, color = "green")

## x軸の範囲は[0, 6]
plt.xlabel("x")
plt.ylabel("y")
plt.show()