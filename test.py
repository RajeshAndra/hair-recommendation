import pandas as pd
import random
data=pd.read_excel("Hairstyle dataset.xlsx")
result=data[(data["Face Shape"]=="Heart") & (data["Gender"]=="Male")]["Hairstyle"]
random.shuffle(list(result))
print(result)