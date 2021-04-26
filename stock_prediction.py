import csv
import numpy as np
COMBO_DATA_PATH = 'data/combo.csv'
from sklearn import datasets, neighbors, linear_model, svm
import pandas as pd
import re

def knn():
    with open(COMBO_DATA_PATH, newline='', encoding='utf-8') as combo:
        reader = csv.DictReader(combo)
        X = []
        y = []
        for row in reader:
            #determine what to include in the feature set
            feature = []
            feature.append(float(row['Post_Num']))

            feature.append(round(float(row['Avg_Upvote_Ratio']), 3))
            keyword = row['Keyword']
            keyword = keyword.strip("[]")
            for i in keyword.split(' '):
                #print(i)
                if (i.isdigit()):
                    feature.append(float(i))
        
            sentiment = row['Sentiment']
            sentiment = sentiment.strip("[]")
            print(sentiment.split(', '))
            for i in sentiment.split(', '):
                # if (i.isdigit()):
                    feature.append(float(i))
            print(feature)

            X.append(feature)
            y.append(int(row['Label']))
        
        print(X[0])
        n_samples = len(X)
        partition_ratio = 0.7 # percentage of the traning set
        print(n_samples)  
        X_train = X[:int(partition_ratio * n_samples)]
        y_train = y[:int(partition_ratio * n_samples)]
        X_test = X[int(partition_ratio * n_samples):]
        y_test = y[int(partition_ratio * n_samples):]

        knn = neighbors.KNeighborsClassifier(n_neighbors=9)
        svm_clf = svm.SVC()
        print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
        print('SVM score: %f' % svm_clf.fit(X_train, y_train).score(X_test, y_test))

    
def test():
    samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(samples)
    print(neigh.kneighbors([[1., 1., 1.]]))

def main():
    #test()
    knn()

if __name__ == "__main__":
    main()