import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# データを取得
# 対象のサイトURL
url = "https://p-town.dmm.com/machines/4676"

# URLリソースを開く
res = urllib.request.urlopen(url)

# インスタンスの作成
soup = BeautifulSoup(res, 'html.parser')

# 必要な要素とclass名
data = soup.find_all("tbody")

# 取得したデータを出力
data_list = []
for data_text in data:
  data_list.append(data_text.text)

# list内の改行と空白を削除
for i in range(len(data_list)):
  data_list[i] = data_list[i].replace('\n','')
  data_list[i] = data_list[i].replace(' ','')

# 標準リストをNumPy配列に変換
ndarray_data = np.array(data_list)

# DataFrameに変換
hoko_df = pd.DataFrame(ndarray_data)

# csvファイルに書き出し
hoko_df.to_csv(r".\data\rise_slot.csv")
