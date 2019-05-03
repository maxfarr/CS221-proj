# import libraries
import numpy as np
import math
import collections
from bs4 import BeautifulSoup
import requests
import pandas as pd

df_popular = pd.DataFrame.from_csv("popular_quotes.csv")
possible_tags = set()
for t in df_popular.tags:
    tags = [x.strip()[1:-1] for x in t[1:-1].split(",")]
    for tag in tags:
        possible_tags.add(tag)
url = "https://www.goodreads.com/quotes/tag/{}?page={}"

num = 1
list_of_df = []
for tag in possible_tags:
    page = requests.get(url.format(tag, 1))
    soup = BeautifulSoup(page.text, 'html.parser')
    container = soup.body.find(class_="mainContentFloat").find(class_ = "leftContainer")
    if(container.find("div", attrs={"style": "float: right;"}).div == None):
        lastpage = 1
    else:
        lastpage = container.find("div", attrs={"style": "float: right;"}).div.contents[-3].contents[0]
        lastpage = int(lastpage)
    print("{}, {}: {}, {} pages".format(num, float(num) / len(possible_tags), tag, lastpage))
    num += 1
    quotefeatures = []
    for i in range(lastpage):
        page = requests.get(url.format(tag, i))
        soup = BeautifulSoup(page.text, 'html.parser')
        container = soup.body.find(class_="mainContentFloat").find(class_ = "leftContainer")
        quotes = container.find(class_="quotes")
        if(quotes == None):
            continue
        quotes = quotes.find_all(class_="quote")
        for quote in quotes:
            text = quote.find("div").find("div").contents[0].strip()[1:-1]
            author = quote.find("div").find("div").find("span").contents[0].strip()
            likes = quote.find("div").find(attrs={"class": "quoteFooter"}).\
                    find("div", attrs={"class", "right"}).find("a").contents[0][:-6]
            likes = int(likes)
            tags = quote.find("div").find(attrs={"class": "quoteFooter"}).find("div").find_all("a")
            taglist = [tag.contents[0] for tag in tags]
            features = {"text": text, "author": author, "likes": likes, "tags": taglist}
            quotefeatures.append(features)
    list_of_df.append(pd.DataFrame(quotefeatures))
big_df = pd.concat(list_of_df)
big_df.drop_duplicates(inplace=True)
big_df.to_csv("tagged_quotes.csv", index_label=False)
