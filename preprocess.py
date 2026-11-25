# For preprocessing data

import pandas as pd
import numpy as np

df = pd.read_csv("IPO_Underpricing_Final.csv")
mapping = {val: i for i, val in enumerate(df['Underpricing_Class'].unique())}

df['Underpricing_Class'] = df['Underpricing_Class'].map(mapping)

df.to_csv("IPO_Underpricing_Final.csv", index=True)
