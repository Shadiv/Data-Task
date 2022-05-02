import pandas as pd 
import numpy as np

s = pd.read_csv('model_data.csv')
df = pd.DataFrame(s)
df['exited'].replace('t',1)
print(df.dtypes)
print(df)