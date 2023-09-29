# 必要なライブラリをimport
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

# 横成分z，縦成分Xの行列
input_matrix = np.array([[1, 1, 1], [1, 1, -1], [1,-1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]])

## 引数: 確率変数の行列
def calc_innerProduct(input_vector):

  # 表を作成する材料
  result = []

  for i in range(3):

    # 行ごとに計算結果を記録
    result_row = []

    for j in range(3):
      inner_x = np.dot(input_vector.T[i],input_vector.T[j]) // 8
      result_row.append(inner_x)

    # 行ごとに計算結果を記録
    result.append(result_row)

  # 表作成
  df = pd.DataFrame(result)

  # DataFrameのタイトルを設定
  df.columns = ['X1', 'X2', 'X3']
  df.index = ['X1', 'X2', 'X3']

  print(df)

## 引数: 確率変数の行列，描画したい行数
def plot_rv(input_vector, number):

  # 描画セット
  fig, ax = plt.subplots()
  ax.plot(np.array([1,2,3]), input_vector[number])
  ax.set_ylim(-2, 2)

  ## 目盛りを整数値にする
  ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
  ax.get_yaxis().set_major_locator(MaxNLocator(integer=True))

  plt.show()

if __name__ == "__main__":
  calc_innerProduct(input_matrix)
  plot_rv(input_matrix, 1)  # 第2引数：0~7の整数