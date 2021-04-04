import csv

GME_STOCK_DATA_PATH = 'data/GME.csv'

# Data preparation for stock price
with open(GME_STOCK_DATA_PATH, newline='') as csvfile:
     reader = csv.DictReader(csvfile)
     for row in reader:
        print(row['Date'], row['Open'], row['Close'])