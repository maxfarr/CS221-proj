import numpy as np
import pandas as pd
import string

def clean_csv(csv_file):
    df = pd.read_csv(csv_file)
    df = df[["author", "likes", "tags", "text"]]
    tags = df["tags"].map(lambda x: x[1:-1])
    df["tags"] = tags
    lower_text = df["text"].astype(str).map(lambda s: " ".join(s.lower().strip().\
                                                               translate(str.maketrans('', '', string.punctuation)).split()))
    df["text"] = lower_text
    lower_text = df["author"].astype(str).map(lambda s: " ".join(s.lower().strip().\
                                                               translate(str.maketrans('', '', string.punctuation)).split()))
    df["author"] = lower_text
    def fn(x):
        m = max([ord(c) for c in x])
        return m < 319
    mask = df["text"].apply(fn)
    df = df[mask]
    df = df.groupby("text").first().reset_index()[["author", "text", "likes", "tags"]]
    df = df.to_csv("{}_clean.csv".format(csv_file[:-4]), index=False)
