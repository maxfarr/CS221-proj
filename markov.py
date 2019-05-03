import numpy as np
from collections import defaultdict
import random

def learnCounts():
    counts = {}
    for string in strings:
        words = string.split()
        for i in range(len(words) - 1):
            if words[i] not in counts:
                counts[words[i]] = (0, defaultdict(int))
            
            counts[words[i]][0] += 1
            counts[words[i]][1][words[i + 1]] += 1

    return counts

def generate(counts, n = 10):
    words = [np.random.choice(counts)]

    while counts[words[0]][0] == 0: words[0] = np.random.choice(counts)
    for i in range(n):
        value = random.randint(0, counts[words[i]] - 1)
        position = 0
        nextWord = ""

        for possible in counts[words[i]][1]:
            position += counts[words[i]][1][possible]
            if value < position:
                nextWord = possible

        words.append(nextWord)

    return words

def main():
    counts = learnCounts()
    sentence = generate(counts)
    print(sentence)