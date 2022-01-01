from numpy.random import RandomState
import pandas as pd

df = pd.read_csv('./datasets/cup/ML-CUP21-TR.csv', sep=",", skiprows=7)
rng = RandomState()
train = df.sample(frac=0.8, random_state=rng)

test = df.loc[~df.index.isin(train.index)]

train.to_csv('train.csv')
test.to_csv('internaltest.csv')
