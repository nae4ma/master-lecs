## 問5~7: ノイズ付きsin関数に関するプログラム
import numpy as np
import matplotlib.pyplot as plt

print("問5")
## サンプルデータD を入力
X = [0.349526784, 1.6974435, 5.384308891, 2.044150596,
     4.578814506, 3.241690807, 2.535931731, 2.210580888,
     3.397474351, 5.972933146, 5.114704101]

Y = [0.254020646, 0.790556868, -0.81239532, 1.012143475,
     -0.904558188, -0.167456361, 0.482547054, 0.878514378,
     -0.210093715, -0.128786937, -0.866501299]

## np.array 変換
noize_x = np.array(X)
noize_y = np.array(Y)

## sin関数を，0.1刻みで計算
x = np.linspace(0, 6, 100)
y = np.sin(x)

## 散布図・グラフを描画
plt.scatter(noize_x, noize_y, color="red")
plt.plot(x, y, color = "green")

## x軸の範囲は[0, 6]
plt.xlabel("x")
plt.ylabel("y")
plt.show()