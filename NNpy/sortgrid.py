import numpy as np
import pandas as pd
import itertools
import numpy as np
import network as nn
import pandas as pd
# print(wi.xavier_init(2,2)[0:, 1:])
import activation_functions as af
import losses
import metrics
from input_reading import read_monk, read_cup
from sklearn.model_selection import train_test_split
"""

train_data, train_labels, valid_data, valid_labels = read_cup(frac_train=0.8)
print(train_data.head())


"""
df = pd.read_csv("grid_-583584758.csv")
# the .sort_values method returns a new dataframe, so make sure to
# assign this to a new variable.
sorted_df = df.sort_values(by=["average_accuracy_vl"], ascending=True)
# Index=False is a flag that tells pandas not to write
# the index of each row to a new column. If you'd like
# your rows to be numbered explicitly, leave this as
# the default, True
sorted_df.to_csv('grid_roberto_2.csv', index=False)
