import matplotlib.pyplot as plt
import csv
from datetime import datetime
import matplotlib.dates as mdates

# plot GME stock price
def plotStockSummary(filename, savepath):
    file = open(filename)
    reader = csv.DictReader(file)
    date = []
    close_price = []
    for row in reader:
        print (row)
        date.append(datetime.strptime(row['Date'], "%Y-%m-%d"))
        close_price.append(float(row['Close']))
    plt.figure()
    plt.plot(date, close_price)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.xlabel("time")
    plt.ylabel("stock price")
    plt.title("GME Stock Price")
    plt.gcf().autofmt_xdate()
    plt.savefig(savepath)
    plt.show()

# plot GME reddit post number and sentiment scores
def plotMetrics(filename):
    file = open(filename)
    reader = csv.DictReader(file)
    post_number = []
    sentiments = []
    dates = []
    for row in reader:
        print (row)
        try:
            dates.append(datetime.strptime(row['Date'], "%Y-%m-%d"))
            post_number.append(int(row['Post_Num']))
            compound = float(row['Sentiment'][1:-1].split()[0][:-1])
            sentiments.append(compound)
        except Exception as e:
            print (e)

    print(sentiments)
    print(post_number)
    # post number picture
    plt.plot(dates, post_number)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.xlabel("time")
    plt.ylabel("post number")
    plt.title("GME Reddit Post Number")
    plt.savefig("img/GME_post_number.png")
    plt.show()

    plt.plot(dates, sentiments)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.xlabel("time")
    plt.ylabel("sentiment")
    plt.title("GME Reddit Sentiments")
    plt.savefig("img/GME_news_sentiments.png")
    plt.show()

def main():
    GME_3_month = "data/GME.csv"
    GME_1_year = "data/GME_1.csv"
    combo = "data/combo.csv"
    plotStockSummary(GME_3_month,"img/GME_3_month.png")
    plotStockSummary(GME_1_year,"img/GME_1_year.png")
    plotMetrics(combo)

if __name__ == "__main__":
    main()