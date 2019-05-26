# import libraries
import collections
from bs4 import BeautifulSoup
import requests
import pandas as pd

url = "https://www.goodreads.com/quotes/tag/{}?page={}"

def process(data, session):
        page = session.get(url.format(data[1], 1))
        if page.status_code != 200:
            return page.status_code
        soup = BeautifulSoup(page.text, 'html.parser')
        container = soup.body.find(class_="mainContentFloat").find(class_ = "leftContainer")
        if(container.find("div", attrs={"style": "float: right;"}).div == None):
                lastpage = 1
        else:
                lastpage = container.find("div", attrs={"style": "float: right;"}).div.contents[-3].contents[0]
                lastpage = int(lastpage)

        print("{} | {}, {} pages".format(data[0], data[1], lastpage))
        
        quotefeatures = []
        for i in range(lastpage):
                page = session.get(url.format(data[1], i))
                if page.status_code != 200:
                    return page.status_code
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
        df = pd.DataFrame(quotefeatures)

        return (df, data[1])