# Stock Prediction Using the Reddit Posts
This is the final project for NYU Big Data Science course 2021 Spring.
# First Steps
1. Git clone this repository
2. Download the datasets: stock and reddit posts.
3. Install required packages using pip or conda
   [Install Packages](https://pip.pypa.io/en/latest/user_guide/#requirements-files))

# Stock Market Prediction Using the Reddit Post Dataset
1. Put the data files under data/ folder
2. Data Preparation: uncomment and run the relavant part of **data_preparation.py**
- preprocess the stock market dataset: stock_preparation()
- preprocess the reddit dataset and then build the features: reddit_feature_extraction()
- integrate the two preprocessed dataset (stock and reddit posts) for later model training: data_integration()
- a function for plotting top keywords for specific time period: keyword_plot()
3. Descriptive Analysis, codes are in **descriptive.py**, the result graphs will be generated in the data/ folder.
4. Modeling, training and evaluation, codes are in **stock_prediction.py**
Three models (logistic regression, random forest and kNN) are used, parameters can later be changed and fine-tuned. Run the main() function you will get the accuracy printed and some evaluation results images under the img/ folder.

# Extension: Arima model
1. Codes are in **arima.py**
2. The model result graphs will be generated in the data/ folder.