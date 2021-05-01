import csv
import numpy as np
from sklearn import datasets, neighbors, linear_model, svm
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

COMBO_DATA_PATH = 'data/combo.csv'

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
        

        # Apply Feature Selection algorithm to select top features if needed
        # X = SelectKBest(k=8).fit_transform(X, y)
        # print(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

        # Evaluate the best n for KNN classifier
        # for i in range(30):
        #     knn = neighbors.KNeighborsClassifier(n_neighbors=i+1)
        #     print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test) + ' n = ' + str(i+1))

        knn = neighbors.KNeighborsClassifier(n_neighbors=12)
        print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test) + ' n = 12')

        lr = LogisticRegression(random_state=30)
        print('Linear regression score: %f' % lr.fit(X_train, y_train).score(X_test, y_test))
        
        rfc = RandomForestClassifier(random_state=30)
        print('Random Forest score: %f' % rfc.fit(X_train, y_train).score(X_test, y_test))
        
        # svm_clf = svm.SVC()
        # print('SVM score: %f' % svm_clf.fit(X_train, y_train).score(X_test, y_test))
        plt.figure()
        plot_confusion_matrix(rfc, X_test, y_test)  
        # plt.show()
        plt.title("Random Forest Confusion Matrix")
        plt.savefig("rf_cm.jpg")
        plt.close()

        plt.figure()
        plot_confusion_matrix(lr, X_test, y_test)  
        # plt.show()
        plt.title("Logistic Regression Confusion Matrix")
        plt.savefig("lr_cm.jpg")
        plt.close()

        plt.figure()
        plot_confusion_matrix(knn, X_test, y_test)  
        # plt.show()
        plt.title("KNN Confusion Matrix")
        plt.savefig("knn_cm.jpg")
        plt.close()

        #knn = neighbors.KNeighborsClassifier(n_neighbors=11)
        #eval(knn, X, y), knn is not suitable for SelectFromModel
        eval(lr, X, y)
        eval(rfc, X, y)

def eval(clf, X, y):
    model = SelectFromModel(clf, prefit=True)
    X = model.transform(X)
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
    # score = cross_val_score(clf, X, y, cv=5)
    print(str(clf) + ' score: %f' % clf.fit(X_train, y_train).score(X_test, y_test))
    plt.figure()
    plot_roc_curve(clf, X_test, y_test)
    plt.title(str(clf) + " ROC")
    plt.savefig(str(clf) + "_roc.jpg")
    plt.close()


def main():
    #test()
    model()

if __name__ == "__main__":
    main()

