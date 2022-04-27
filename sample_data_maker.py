# imports
import numpy as np
import pandas as pd

data = pd.read_csv("le_cleaned.csv")
data.sort_values('Country', inplace=True)

data_x = data.loc[:,data.columns != 'Life_expectancy']
data_y = data['Life_expectancy']

data_x.to_csv("input.csv", index=False)
data_y.to_csv("labels.csv", index=False)