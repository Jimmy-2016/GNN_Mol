
import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/bace.csv').reset_index()

pp = 0.66

index = np.random.permutation(len(df))
df = df.iloc[index, :]

df_train = df.iloc[:int(pp*len(df)), :]
df_test = df.iloc[int(pp*len(df)):, :]

df_train.to_csv('./data/raw/train_bace.csv', index=False)
df_test.to_csv('./data/raw/test_bace.csv', index=False)


