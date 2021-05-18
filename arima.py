import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import lag_plot
import matplotlib.dates as mdates
import pmdarima as pm

# plot autocorrelation and order differencing graph
def plotMetrics(filename, savepath):
    df = pd.read_csv(filename)
    print(df.head())
    plt.figure()
    plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

    lag_plot(df['Close'], lag=1)
    plt.title('GME Stock - Autocorrelation plot with lag = 1')
    plt.savefig(savepath+"_ac_lag_1.png")
    plt.show()

    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(df.Close);
    axes[0, 0].set_title('Original Series')
    plot_acf(df.Close, ax=axes[0, 1])
    # 1st Differencing
    axes[1, 0].plot(df.Close.diff());
    axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df.Close.diff().dropna(), ax=axes[1, 1])

    # 2nd Differencing
    axes[2, 0].plot(df.Close.diff().diff());
    axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df.Close.diff().diff().dropna(), ax=axes[2, 1])
    plt.savefig(savepath+"_metrics.png")
    plt.show()

# train ARIMA model on the training dataset
def trainModel(GME_3_month, GME_1_year):
    df_3_month = pd.read_csv(GME_3_month)
    df_1_year = pd.read_csv(GME_1_year)

    # GME 3 months
    auto_model = pm.auto_arima(df_3_month.Close, start_p=1, start_q=1,
                               information_criterion='aic',
                               test='adf',  # use adftest to find optimal 'd'
                               max_p=3, max_q=3,  # maximum p and q
                               m=1,  # frequency of series
                               d=None,  # let model determine 'd'
                               seasonal=False,  # No Seasonality
                               start_P=0,
                               D=0,
                               trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)
    print(auto_model.summary)
    # 1,1,0 ARIMA Model
    model = ARIMA(df_3_month.Close, order=(1, 1, 0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # Plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(2, 1)
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.savefig("img/GME_3_month_residuals.png")
    plt.show()
    # Actual vs Fitted
    model_fit.plot_predict(dynamic=False)
    plt.savefig("img/GME_3_month_fit.png")
    plt.show()

    index = int(len(df_3_month.Close) * 0.95)

    train = df_3_month.Close[:index]
    test = df_3_month.Close[index:]
    prediction(train,test, 1, 1, 0, "GME_3_month_prediction.png")

    # GME 1 year
    auto_model = pm.auto_arima(df_1_year.Close, start_p=1, start_q=1,
                               information_criterion='aic',
                               test='adf',  # use adftest to find optimal 'd'
                               max_p=10, max_q=10,  # maximum p and q
                               m=1,  # frequency of series
                               d=None,  # let model determine 'd'
                               seasonal=False,  # No Seasonality
                               start_P=0,
                               D=0,
                               trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)
    print(auto_model.summary)
    # 5,1,0 ARIMA Model
    model = ARIMA(df_1_year.Close, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # Plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(2, 1)
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.savefig("img/GME_1_year_residuals.png")
    plt.show()
    # Actual vs Fitted
    model_fit.plot_predict(dynamic=False)
    plt.savefig("img/GME_1_year_fit.png")
    plt.show()

    index = int(len(df_1_year.Close) * 0.95)

    train = df_1_year.Close[:index]
    test = df_1_year.Close[index:]
    prediction(train,test, 5, 1, 0, "GME_1_year_prediction.png")

# test ARIMA model on the testing dataset
def prediction(train, test, p, d, q, savepath):
    # Build Model
    model = ARIMA(train, order=(p, d, q))
    fitted = model.fit(disp=-1)

    # Forecast
    fc, se, conf = fitted.forecast(len(test), alpha=0.05)  # 95% conf

    # Make as pandas series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)

    # Plot
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.savefig(savepath)
    plt.show()

def main():
    GME_3_month = "data/GME.csv"
    GME_1_year = "data/GME_1.csv"
    plotMetrics(GME_3_month, "img/GME_3_month")
    plotMetrics(GME_1_year, "img/GME_1_year")
    trainModel(GME_3_month,GME_1_year)

if __name__ == "__main__":
    main()