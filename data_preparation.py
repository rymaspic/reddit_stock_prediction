import csv
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import re

RAW_GME_STOCK_DATA_PATH = 'data/GME.csv'
PROCESSED_GME_STOCK_DATA_PATH = 'data/GME_processed.csv'

RAW_REDDIT_DATA_PATH = 'data/submissions_reddit.csv'
PROCESSED_REDDIT_DATA_PATH = 'data/submissions_reddit_processed.csv'

def stock_preparation():
   # Data preparation for stock price
   with open(RAW_GME_STOCK_DATA_PATH, newline='') as csvfile, open(PROCESSED_GME_STOCK_DATA_PATH, mode='w') as csv_output_file:
     reader = csv.DictReader(csvfile)
     fieldnames = ['Date', 'Open', 'Close', 'Volumn', 'Label', 'Percentage']
     writer = csv.DictWriter(csv_output_file, fieldnames=fieldnames)
     writer.writeheader()
     for row in reader:
        date = row['Date']
        open_price = row['Open']
        close_price = row['Close']
        volume = row['Volume']
        percentage = (float(close_price) - float(open_price))/float(open_price)
        #label = 1, stock price rises that day, label = 0, price decreses that day
        label = 1 if (float(close_price) - float(open_price) > 0) else 0
        writer.writerow({'Date': date, 'Open': open_price, 'Close': close_price, 'Volumn': volume, 'Label': label, 'Percentage': percentage})
        #print(date, open_price, close_price, label)

def reddit_feature_extraction():
   with open(RAW_REDDIT_DATA_PATH, newline='') as csvfile, open(PROCESSED_REDDIT_DATA_PATH, mode='w') as csv_output_file:
     reader = csv.DictReader(csvfile)
     fieldnames = ['Date', 'Title', 'Post_Num', 'Avg_Upvote_Ratio', 'Keyword', 'Sentiment'] #todo: keyword and sentiment features
     writer = csv.DictWriter(csv_output_file, fieldnames=fieldnames)
     writer.writeheader()
     post_count = 0
     upvote_radio_count = 0
     document_list = []
     prev_date = '2021-01-01'
     for row in reader:
        date = row['created'].split()[0]
        title = row['title']
        upvote_ratio = row['upvote_ratio']
        print(date, title)
        if (date == prev_date):
           document_list.append(title)
           post_count = post_count + 1
           upvote_radio_count = upvote_radio_count + float(upvote_ratio)
        else:
           keyword_features = keyword_feature(document_list)
           writer.writerow({'Date': prev_date, 'Title': document_list, 'Post_Num': post_count, 'Avg_Upvote_Ratio': upvote_radio_count/post_count, 'Keyword': keyword_features, 'Sentiment': []})
           post_count = 1
           upvote_radio_count = float(upvote_ratio)
           document_list = [title]
           prev_date = date

#todo-1
# input is a the document_list object, a list of string. eg. ['gme to the moon', 'buy and hold', 'lets go gme gang 🚀']
# output a 1*n matrix with the count of frequency of the target words each day. eg. [0, 1, 2, 0, 0, 0, 1]
def keyword_feature(list_of_sentences):
   corpus = list_of_sentences
   vocabulary = ['moon', 'like', 'buy', 'hold', 'apes']
   emoji = ['\U0001F680']
   f1 = CountVectorizer(vocabulary=vocabulary).fit_transform(corpus).toarray()
   f2 = CountVectorizer(vocabulary=emoji, analyzer='char', binary=True).fit_transform(corpus).toarray() #for emoji, we count using one-hot
   output = np.concatenate((f1, f2), axis=1)
   output = output.sum(axis = 0)
   return output

def preprocess_text(text):
   text = re.sub(r'[0-9]+', ' ', text)
   return text

# function to count the top keywords generated from all reddit posts
def keyword_count():
   with open(RAW_REDDIT_DATA_PATH, newline='') as csvfile:
     reader = csv.DictReader(csvfile)
     corpus = []
     for row in reader:
        corpus.append(row['title'])
     print("start term frequency analysis...")
     #target_vocabulary = ['moon', 'yolo', 'buy', 'hold', 'rise']
     #corpus = ["moon the moon the hello", "I love gme", "go to moon"]
     cv = CountVectorizer(analyzer='word', lowercase=True, stop_words='english')
     cv_fit = cv.fit_transform(corpus)
     counts = pd.DataFrame(cv_fit.toarray(),columns=cv.get_feature_names())
     counts = counts.sum()
     print(counts.sort_values(ascending=False).head(10))
     print("start tfidf analysis ...")
     tfIdfVectorizer = TfidfVectorizer(use_idf=True,lowercase=True, stop_words='english')
     tfIdf = tfIdfVectorizer.fit_transform(corpus)
     counts_tf = pd.DataFrame(tfIdf.toarray(),columns=tfIdfVectorizer.get_feature_names())
     counts_tf = counts_tf.sum()
     print(counts_tf.sort_values(ascending=False).head(10))
     
#todo-2
#input is a the document_list object, a list of string. eg. ['gme to the moon', 'buy and hold', 'lets go gme gang 🚀']
def sentiment_feature(list_of_sentences):
   return[]

def main():
    #keyword_count()
    reddit_feature_extraction()

if __name__ == "__main__":
    main()