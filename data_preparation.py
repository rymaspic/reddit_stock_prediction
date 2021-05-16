import csv
import sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from datetime import datetime as dt

csv.field_size_limit(2147483647)
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

RAW_GME_STOCK_DATA_PATH = 'data/GME.csv'
PROCESSED_GME_STOCK_DATA_PATH = 'data/GME_processed.csv'

RAW_REDDIT_DATA_PATH = 'data/submissions_reddit.csv'
PROCESSED_REDDIT_DATA_PATH = 'data/submissions_reddit_processed.csv'
TEST_REDDIT_DATA_PATH = 'data/reddit_test.csv'

COMBO_DATA_PATH = 'data/combo.csv'
nltk.download([
     "names",
     "stopwords",
     "averaged_perceptron_tagger",
     "punkt",
     'vader_lexicon',
])
stopwords = nltk.corpus.stopwords.words("english")

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
   with open(RAW_REDDIT_DATA_PATH, newline='', encoding='utf-8') as csvfile, open(PROCESSED_REDDIT_DATA_PATH, mode='w', encoding='utf-8') as csv_output_file:
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
        #print(date, title)
        if (date == prev_date):
           document_list.append(title)
           post_count = post_count + 1
           upvote_radio_count = upvote_radio_count + float(upvote_ratio)
        else:
           keyword_features = keyword_feature(document_list)
           sentiment_features = sentiment_feature(document_list)
           writer.writerow({'Date': prev_date, 'Title': document_list, 'Post_Num': post_count, 'Avg_Upvote_Ratio': upvote_radio_count/post_count, 'Keyword': keyword_features, 'Sentiment': sentiment_features})
           post_count = 1
           upvote_radio_count = float(upvote_ratio)
           document_list = [title]
           prev_date = date

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
   with open(RAW_REDDIT_DATA_PATH, newline='', encoding='utf-8') as csvfile:
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
     print(counts.sort_values(ascending=False).head(20))
     print("start tfidf analysis ...")
     tfIdfVectorizer = TfidfVectorizer(use_idf=True,lowercase=True, stop_words='english')
     tfIdf = tfIdfVectorizer.fit_transform(corpus)
     counts_tf = pd.DataFrame(tfIdf.toarray(),columns=tfIdfVectorizer.get_feature_names())
     counts_tf = counts_tf.sum()
     print(counts_tf.sort_values(ascending=False).head(20))

#funtion to plot the top keywords of the given time period
def keyword_plot(date_start, date_end, n):
   with open(RAW_REDDIT_DATA_PATH, newline='', encoding='utf-8') as csvfile:
     reader = csv.DictReader(csvfile)
     corpus = []
     dt_start = dt.strptime(date_start, '%Y-%m-%d')
     dt_end = dt.strptime(date_end, '%Y-%m-%d')
     for row in reader:
        date = dt.strptime(row['created'].split()[0], '%Y-%m-%d')
         #   print(date)
         #   print(dt_start)
         #   print(dt_end)
         #   print(date >= dt_start)
        if ((date >= dt_start) and (date <= dt_end)):
            corpus.append(row['title'])
      
     print("start term frequency analysis...")
     cv = CountVectorizer(analyzer='word', lowercase=True, stop_words='english')
     cv_fit = cv.fit_transform(corpus)
     counts = pd.DataFrame(cv_fit.toarray(),columns=cv.get_feature_names())
     counts = counts.sum()
     counts = counts.sort_values(ascending=False).head(n)
     print(counts)
     print(counts.index.values)
     x = counts.index.values
     y = []
     for i in x:
        print(counts[i])
        y.append(counts[i])
     plt.figure()
     plt.bar(x = x, height = y)
     title = "Top-" + str(n) + " keyword count from "+date_start+" to "+date_end 
     plt.title(title)
     plt.savefig("img/keyword_count.jpg")
     plt.close()

     print("start tfidf analysis ...")
     tfIdfVectorizer = TfidfVectorizer(use_idf=True,lowercase=True, stop_words='english')
     tfIdf = tfIdfVectorizer.fit_transform(corpus)
     counts_tf = pd.DataFrame(tfIdf.toarray(),columns=tfIdfVectorizer.get_feature_names())
     counts_tf = counts_tf.sum()
     counts_tf = counts_tf.sort_values(ascending=False).head(n)
     print(counts_tf)

     x = counts_tf.index.values
     y = []
     for i in x:
        print(counts_tf[i])
        y.append(counts_tf[i])
     plt.figure()
     plt.bar(x = x, height = y)
     title = "Top-" + str(n) + " keyword using TF-IDF from "+date_start+" to "+date_end 
     plt.title(title)
     plt.savefig("img/Tfidf_count.jpg")
     plt.close()

#input is a the document_list object, a list of string. eg. ['gme to the moon', 'buy and hold', 'lets go gme gang 🚀']
#output is a 1*4 matrix with the sentiment score [compound, pos, neu, neg]
#if compound value > 0, it means positive
def sentiment_feature(list_of_sentences):
    text = " ".join(list_of_sentences)
    tokens = nltk.word_tokenize(text)
    tokens_without_sw = [token for token in tokens if token.lower() not in stopwords]
    sia = SentimentIntensityAnalyzer()
    txt = " ".join(tokens_without_sw)
    sentiment_scores = sia.polarity_scores(txt)
    output = [sentiment_scores['compound'], sentiment_scores['pos'], sentiment_scores['neu'], sentiment_scores['neg']]
    print(output)
    return output


# function to integrate the processed stock dataset and reddit post dataset
def data_integration():
   with open(PROCESSED_GME_STOCK_DATA_PATH, newline='', encoding='utf-8') as gme, open(PROCESSED_REDDIT_DATA_PATH, newline='', encoding='utf-8') as feature, open(COMBO_DATA_PATH, mode='w', encoding='utf-8') as csv_output_file:
      reader_gme = csv.DictReader(gme)
      reader_feature = csv.DictReader(feature)
      fieldnames = ['Date', 'Post_Num', 'Avg_Upvote_Ratio', 'Keyword', 'Sentiment', 'Label']
      writer = csv.DictWriter(csv_output_file, fieldnames=fieldnames)
      writer.writeheader()
      stock_dict = {}
      stock_date_list = []
      for i in reader_gme:
         date_time_obj = dt.strptime(i['Date'], '%Y-%m-%d')
         print(date_time_obj)
         print(i['Label'])
         stock_dict[date_time_obj] = i['Label']
         stock_date_list.append(date_time_obj)
      
      for row in reader_feature:
         # the reddit post date
         date = dt.strptime(row['Date'], '%Y-%m-%d')
         # the relevant stock date
         stock_date = min(stock_date_list, key=lambda d: abs(d - date))
         print(date)
         print("nearest stock market date",date)
         date_string = str(date.year)+ "-" + str(date.month) + "-" + str(date.day)
         print(date_string)
         # choose the features that we decide to use
         writer.writerow({'Date': date_string, 'Keyword': row['Keyword'], 'Post_Num': row['Post_Num'], 'Avg_Upvote_Ratio': row['Avg_Upvote_Ratio'], 'Sentiment': row['Sentiment'], 'Label': stock_dict[stock_date]})

    
def main():
    #step1: preprocessing the stock price dataset, input dataset name:"GME.csv", output dataset name: "GME_processed.csv"
    stock_preparation()
    #step2: extract the features from the reddit dataset, input dataset name: "submissions_reddit.csv", output dataset name: "submissions_reddit_processed.csv"
    reddit_feature_extraction()
    #step3: integrate the two dataset into one which can be used for model training, output dataset name: "combo.csv"
    #data_integration()
    #descriptive analysis: plot the keywords in a given time period
    #keyword_plot("2021-01-01", "2021-03-31", 10)

if __name__ == "__main__":
    main()