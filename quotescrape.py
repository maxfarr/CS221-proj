# import libraries
import numpy as np
import math
import collections
from bs4 import BeautifulSoup
import requests
import pandas as pd
#from numba import jit
from multiprocessing import Pool
from tqdm import tqdm

url = "https://www.goodreads.com/quotes/tag/{}?page={}"

def process(data):
        page = requests.get(url.format(data[1], 1))
        soup = BeautifulSoup(page.text, 'html.parser')
        container = soup.body.find(class_="mainContentFloat").find(class_ = "leftContainer")
        if(container.find("div", attrs={"style": "float: right;"}).div == None):
                lastpage = 1
        else:
                lastpage = container.find("div", attrs={"style": "float: right;"}).div.contents[-3].contents[0]
                lastpage = int(lastpage)

        print("{} | {}, {} pages".format(data[0], data[1], lastpage))
        
        #pbar = tqdm(total=lastpage, leave=False)
        quotefeatures = []
        for i in range(lastpage):
                page = requests.get(url.format(data[1], i))
                soup = BeautifulSoup(page.text, 'html.parser')
                container = soup.body.find(class_="mainContentFloat").find(class_ = "leftContainer")
                quotes = container.find_all("div", attrs={"class": "quote"})
                for quote in quotes:
                        text = quote.find("div").find("div").contents[0].strip()[1:-1]
                        author = quote.find("div").find("div").find("span").contents[0].strip()
                        likes = quote.find("div").find(attrs={"class": "quoteFooter"}).\
                                find("div", attrs={"class", "right"}).find("a").contents[0][:-6]
                        likes = int(likes)
                        tags = quote.find("div").find(attrs={"class": "quoteFooter"}).find("div").find_all("a")
                        taglist = [tag.contents[0] for tag in tags]
                        taglist = ", ".join(taglist)
                        features = {"text": text, "author": author, "likes": likes, "tags": taglist}
                        quotefeatures.append(features)
                #pbar.update()
        df = pd.DataFrame(quotefeatures)

        return df

if __name__ == '__main__':
        df_popular = pd.DataFrame.from_csv("popular_quotes.csv")
        possible_tags = set()
        for t in df_popular.tags:
                tags = [x.strip()[1:-1] for x in t[1:-1].split(",")]
                for tag in tags:
                        possible_tags.add(tag)

        list_of_df = []

        pbar = tqdm(total=len(possible_tags))

        def add(df):
                list_of_df.append(df)
                pbar.update()

        pool = Pool(8)

        for data in enumerate(possible_tags):
                pool.apply_async(process, args=[data], callback=add)

        pool.close()
        pool.join()

        big_df = pd.concat(list_of_df)
        big_df.drop_duplicates(subset = ["text", "author", "likes", "tags"], inplace=True)
        big_df.to_csv("tagged_quotes.csv", index_label=False)
