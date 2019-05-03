import pandas as pd


df = pd.read_csv("popular_quotes.csv")
df = df.drop('Unnamed: 0', 1)

quotes = df.text