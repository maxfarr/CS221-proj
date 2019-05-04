import numpy as np
from collections import defaultdict
import random
import pandas as pd

def learnCounts(strings):
    counts = {}
    for string in strings:
        if type(string) != str: continue
        words = string.split()
        for i in range(len(words) - 1):
            if words[i] not in counts:
                counts[words[i]] = [0, defaultdict(int)]
            
            counts[words[i]][0] += 1
            counts[words[i]][1][words[i + 1]] += 1

    return counts

def generate(counts, n = 10):
    words = [np.random.choice(list(counts.keys()))]
    #while words[0] not in counts: words[0] = np.random.choice(list(counts.keys()))

    i = 0
    failed = False
    while i < n:
        value = random.randint(1, counts[words[i]][0])
        position = 0
        nextWord = ""

        for possible in counts[words[i]][1]:
            #if value == 1: print(counts[words[i]][1][possible])
            position += counts[words[i]][1][possible]
            if position <= value:
                nextWord = possible

        if((nextWord not in counts or counts[nextWord][0] == 0) and i < n - 1):
            #print(nextWord)
            failed = True
            continue

        words.append(nextWord)
        i += 1

    return words

def main():
    df = pd.read_csv("popular_quotes.csv")
    df = df.drop('Unnamed: 0', 1)
    quotes = df.text.tolist()
    counts = learnCounts(quotes)
    for i in range(15):
        sentence = generate(counts, i)
        print(" ".join(sentence))

if __name__ == "__main__":
    main()