import csv
import numpy as np
COMBO_DATA_PATH = 'data/combo.csv'
from sklearn import datasets, neighbors, linear_model, svm
import pandas as pd
import re
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split

def model():
    with open(COMBO_DATA_PATH, newline='', encoding='utf-8') as combo:
        reader = csv.DictReader(combo)
        X = []
        y = []
        # Features: Post_Num, Avg_Upvote_Ratio, Keyword Counts, Sentiment Scores (12 columns)
        # So far I include all columns from the combo.csv
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
        
        #print(X[0])
        n_samples = len(X)

        #feature selection
        X = SelectKBest(k=8).fit_transform(X, y)
        # print(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

        # knn = neighbors.KNeighborsClassifier(n_neighbors=9)
        for i in range(30):
            knn = neighbors.KNeighborsClassifier(n_neighbors=i+1)
            print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test) + ' n = ' + str(i))
        
        svm_clf = svm.SVC()
        print('SVM score: %f' % svm_clf.fit(X_train, y_train).score(X_test, y_test))

    
def test():
    samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(samples)
    print(neigh.kneighbors([[1., 1., 1.]]))

def main():
    #test()
    model()

if __name__ == "__main__":
    main()