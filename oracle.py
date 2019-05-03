import pandas as pd

if name == "__main__":
    df = pd.read_csv("popular_quotes.csv")
    df = df.drop('Unnamed: 0', 1)

    quotes = df.text.tolist()
    quote = random.choice(quotes)
    print(quote)