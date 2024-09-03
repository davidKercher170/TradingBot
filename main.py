from Algorithms import LinkMomentum, SectorMomentum, MomentumPortfolio, LinkPortfolio
from BacktestMethods import Backtest
from Display import BacktestDisplay

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def link_momentum_test(d,corr,path,interval=250, alpha_val = False):
    # open the data for the correlation matrix
    with open(path+d+corr+"/"+str(interval)+"Day.pkl", 'rb') as handle:
        arr=pd.read_pickle(handle)
        op=pd.read_pickle(path+d+'Open.pkl') # load Open Data
        df=pd.read_pickle(path+d+'Close.pkl') # load Close Data

        # get the first, last keys (dates) and the indices for corresponding keys
        first_key=list(arr.keys())[0]
        last_key=list(arr.keys())[-1]
        start_index=list(str(d.date()) for d in df.index).index(first_key)
        end_index=list(str(d.date()) for d in df.index).index(last_key)

        ticks=list(i for i in op.columns[:int(len(op.columns)/2)]) # stock tickers to look at
        start_date=list(d for d in df.index)[start_index] # first date of trading
        end_date=list(d for d in df.index)[end_index] # last date of trading

        print(start_date)


    lm = LinkMomentum(arr,ticks,k2=1,method="dif",alpha_val=a) # initialize Algorithm
    bt = Backtest(lm,df,op,tickers=ticks,start_date=start_date,end_date=end_date,dates=list(d.date() for d in df.index)[start_index:end_index],meas_stock=[]) # initialize backtest
    bt.backtest() # backtest
    bt.plot(d[:-1])
    # print("Vol: ", bt.calc_vol())
    # print("Sharpe: ", bt.sharpe())

    return bt.calc_vol(), bt.sharpe()


def mlink_momentum_test(dir_list, corr, path, interval=250, alpha_val=False, signal_thresholds=[],num_terms=10):
    op = []
    df = []
    ticks = []
    algs = []
    arrs = []

    sp500 = pd.read_pickle(path+'SP500Open.pkl')
    for d in dir_list:
            open_data = pd.DataFrame()
            close_data = pd.DataFrame()

            # open the data for the correlation matrix
            with open(path+d+corr+"/"+str(interval)+"Day.pkl", 'rb') as handle:
                arr=pd.read_pickle(handle)

            open_data=pd.read_pickle(path+d+'Open.pkl') # load Open Data
            close_data=pd.read_pickle(path+d+'Close.pkl') # load Close Data

            # get the first, last keys (dates) and the indices for corresponding keys
            first_key=list(arr.keys())[0]
            last_key=list(arr.keys())[-1]
            start_index=list(str(d.date()) for d in close_data.index).index(first_key)
            end_index=list(str(d.date()) for d in close_data.index).index(last_key)


            t=list(i for i in open_data.columns[:int(len(open_data.columns)/2)]) # stock tickers to look at

            start_date=list(d for d in open_data.index)[start_index] # first date of trading
            end_date=list(d for d in open_data.index)[end_index] # last date of trading

            dates = list(x.date() for x in close_data.index)[start_index+20:-20]

            ticks.append(t)
            op.append(open_data)
            df.append(close_data)
            arrs.append(arr)
            algs.append(LinkMomentum(arr, t, k2=1,method="dif", alpha_val=alpha_val, num_terms=num_terms))

    # print("Start Date: " + str(dates[0]) + " --- End Date: " + str(dates[-1]))
    bt = Backtest(algs,df,op,tickers=ticks,start_date=start_date,end_date=end_date,dates=dates,meas_stock=sp500, multi_algs=True, signal_thresholds=False) # initialize backtest

    bt.backtest() # backtest
    bt.plot("Combined")
    # e = bt.eval()

    with open("log.txt", "w") as f:
        f.write(bt.log)

    # print(dir_list[0][:-1] + ". Buy: " + str(bt.avg_bsignal[0]) + " --- Sell: " + str(bt.avg_ssignal[0]))
    return bt.calc_vol(), bt.sharpe(), bt.comp_annual(), 0


def momentum_portfolio_test(dir_list, corr, path, interval=250, P=20):
    op = []
    df = []
    ticks = []
    algs = []
    arrs = []

    sp500 = pd.read_pickle(path+'SP500Open.pkl')
    for d in dir_list:
            open_data = pd.DataFrame()
            close_data = pd.DataFrame()

            # open the data for the correlation matrix
            with open(path+d+corr+"/"+str(interval)+"Day.pkl", 'rb') as handle:
                arr=pd.read_pickle(handle)

            open_data=pd.read_pickle(path+d+'Open.pkl') # load Open Data
            close_data=pd.read_pickle(path+d+'Close.pkl') # load Close Data

            # get the first, last keys (dates) and the indices for corresponding keys
            first_key=list(arr.keys())[0]
            last_key=list(arr.keys())[-1]
            start_index=list(str(d.date()) for d in close_data.index).index(first_key)
            end_index=list(str(d.date()) for d in close_data.index).index(last_key)


            t=list(i for i in open_data.columns[:int(len(open_data.columns)/2)]) # stock tickers to look at

            start_date=list(d for d in open_data.index)[start_index] # first date of trading
            end_date=list(d for d in open_data.index)[end_index] # last date of trading

            dates = list(x.date() for x in close_data.index)[start_index+20:-20]

            ticks.append(t)
            op.append(open_data)
            df.append(close_data)
            arrs.append(arr)
            algs.append(MomentumPortfolio(arr, t))

    bt = Backtest(algs,df,op,tickers=ticks,start_date=start_date,end_date=end_date,dates=dates,meas_stock=sp500, multi_algs=True, signal_thresholds=False, P=P) # initialize backtest
    bt.backtest() # backtest
    # bt.plot("Combined")
    # e = bt.eval()

    with open("log.txt", "w") as f:
        f.write(bt.log)

    # print(dir_list[0][:-1] + ". Buy: " + str(bt.avg_bsignal[0]) + " --- Sell: " + str(bt.avg_ssignal[0]))
    return bt.calc_vol(), bt.sharpe(), bt.comp_annual(), 0


def link_portfolio_test(dir_list, corr, path, interval=250, alpha_val=False, signal_thresholds=[],num_terms=10, rank=1):
    op = []
    df = []
    ticks = []
    algs = []
    arrs = []

    sp500 = pd.read_pickle(path+'SP500Open.pkl')
    for d in dir_list:
            open_data = pd.DataFrame()
            close_data = pd.DataFrame()

            # open the data for the correlation matrix
            with open(path+d+corr+"/"+str(interval)+"Day.pkl", 'rb') as handle:
                arr=pd.read_pickle(handle)

            open_data=pd.read_pickle(path+d+'Open.pkl') # load Open Data
            close_data=pd.read_pickle(path+d+'Close.pkl') # load Close Data

            # get the first, last keys (dates) and the indices for corresponding keys
            first_key=list(arr.keys())[0]
            last_key=list(arr.keys())[-1]
            start_index=list(str(d.date()) for d in close_data.index).index(first_key)
            end_index=list(str(d.date()) for d in close_data.index).index(last_key)


            t=list(i for i in open_data.columns[:int(len(open_data.columns)/2)]) # stock tickers to look at

            start_date=list(d for d in open_data.index)[start_index] # first date of trading
            end_date=list(d for d in open_data.index)[end_index] # last date of trading

            dates = list(x.date() for x in close_data.index)[start_index+20:-20]

            ticks.append(t)
            op.append(open_data)
            df.append(close_data)
            arrs.append(arr)
            algs.append(LinkPortfolio(arr, t, k2=1,method="dif", alpha_val=alpha_val, num_terms=num_terms, r=rank))

    # print("Start Date: " + str(dates[0]) + " --- End Date: " + str(dates[-1]))
    bt = Backtest(algs,df,op,tickers=ticks,start_date=start_date,end_date=end_date,dates=dates,meas_stock=sp500, multi_algs=True, signal_thresholds=False) # initialize backtest

    bt.backtest() # backtest
    bt.plot("Combined")
    print(bt.monthly_stats())
    # e = bt.eval()

    with open("log.txt", "w") as f:
        f.write(bt.log)

    # print(dir_list[0][:-1] + ". Buy: " + str(bt.avg_bsignal[0]) + " --- Sell: " + str(bt.avg_ssignal[0]))
    return bt.calc_vol(), bt.sharpe(), bt.comp_annual(), 0


def sector_dif(dir_list, corr, path, interval=250, alpha_val=False, signal_thresholds=[]):
    op = []
    df = []
    ticks = []
    algs = []
    arrs = []

    sp500 = pd.read_pickle(path+'SP500Open.pkl')
    sdf = pd.read_pickle(path+'Sectors/Open.pkl')
    mom_keys = list(i for i in sdf.columns[int(len(sdf.columns)/2):])
    i=-1

    for d in dir_list:
        i+= 1

        # open the data for the correlation matrix
        with open(path+d+corr+"/"+str(interval)+"Day.pkl", 'rb') as handle:
            arr=pd.read_pickle(handle)

        open_data=pd.read_pickle(path+d+'Open.pkl') # load Open Data

        # get the first, last keys (dates) and the indices for corresponding keys
        first_key=list(arr.keys())[0]
        last_key=list(arr.keys())[-1]
        start_index=list(str(d.date()) for d in open_data.index).index(first_key)
        end_index=list(str(d.date()) for d in open_data.index).index(last_key)


        t=list(i for i in open_data.columns[:int(len(open_data.columns)/2)]) # stock tickers to look at

        open_data["Index"] = sdf.iloc[:, 1]
        open_data = open_data.fillna(0)
        mom = sdf[mom_keys[i]]
        t.append("Index")

        start_date=list(d for d in open_data.index)[start_index] # first date of trading
        end_date=list(d for d in open_data.index)[end_index] # last date of trading

        dates = list(x.date() for x in open_data.index)[start_index+20:-20]

        ticks.append(t)
        op.append(open_data)
        arrs.append(arr)
        algs.append(SectorMomentum(arr, t, k2=1,method="dif", alpha_val=alpha_val,sector_momentum=mom))

    # print("Start Date: " + str(dates[0]) + " --- End Date: " + str(dates[-1]))
    bt = Backtest(algs,df,op,tickers=ticks,start_date=start_date,end_date=end_date,dates=dates,meas_stock=sp500,multi_algs=True, signal_thresholds=False,sector_momentum=sdf) # initialize backtest

    bt.backtest() # backtest
    bt.plot("Combined")
    e = bt.eval()

    with open("log.txt", "w") as f:
        f.write(bt.log)

    print(dir_list[0][:-1] + ". Buy: " + str(bt.avg_bsignal[0]) + " --- Sell: " + str(bt.avg_ssignal[0]))
    return bt.calc_vol(), bt.sharpe(), bt.comp_annual(), round(e,3)


def test_1():
    for index in index_list:
        print(index)
        sharpe_vals = []
        cag_vals = []
        for n in [2,16]:
            v, sh, cag, e = mlink_momentum_test([index], "Spearman", "TradingBot/Data/",  alpha_val = a, num_terms=n)
            sharpe_vals.append(sh)
            cag_vals.append(cag)

        print(np.round(sharpe_vals,2))
        print(np.round(cag_vals,2))
        print("----")


def test_2():

    avg_v, avg_sh, avg_cag, avg_w = 0, 0, 0, 0
    for s in index_list:
        v, sh, cag, e = mlink_momentum_test([s], "Spearman", "TradingBot/Data/",  alpha_val = a,num_terms=10)
        avg_v += v/7
        avg_sh += sh/7
        avg_cag += cag/7
        avg_w += e/7
        print(s[:-1] + " --- Volatility: " + str(round(v,2)) + " --- Sharpe: " + str(round(sh,2)) + " --- CAG: " + str(round(cag,2)) + " --- EV: " + str(e))

    print("Average --- Volatility: " + str(round(avg_v,2)) + " --- Sharpe: " + str(round(avg_sh,2)) + " --- CAG: " + str(round(avg_cag,2)) + " --- EV: " + str(avg_w))


a = 0.75
index_list = ["Electric/", "Energy/", "Finance/", "Health/", "Indust/", "REstate/", "Tech/"]
avg_cag = 0
avg_sh = 0

a = 0.75
for n_terms in [1,5,10,15,20]:
    v, sh, cag, e = mlink_momentum_test(index_list, "Spearman", "TradingBot/Data/",  alpha_val = a, num_terms=n_terms, interval=250)
    print("Alpha: " + str(a) + " --- Terms: " + str(n_terms) + " --- Sharpe: " + str(round(sh,2)) + " --- CAG: " + str(round(cag,2)))

# avg_cag = 0
# avg_sh = 0
# print("Momentum Portfolio")
# for index in index_list:
#     v, sh, cag, e = momentum_portfolio_test([index], "Spearman", "TradingBot/Data/", P=20)
#     print("Index: " + index + " --- Sharpe: " + str(round(sh,2)) + " --- CAG: " + str(round(cag,2)))

#     avg_cag += cag
#     avg_sh += sh
# print("Average: " + str(avg_cag/7) + "  " + str(avg_sh/7))


# avg_cag = 0
# avg_sh = 0
# print("Link Portfolio")
# for q in range(5,10):
#     v, sh, cag, e = link_portfolio_test(index_list, "Spearman", "TradingBot/Data/", alpha_val=0.85, num_terms=20, rank=q)
#     print("Rank: " + str(q) + " --- Sharpe: " + str(round(sh,2)) + " --- CAG: " + str(round(cag,2)))


# a = 0.85
# avg_cag = 0
# avg_sh = 0
# print("Strategy: MultiLink")
# for index in index_list:
#     v, sh, cag, e = mlink_momentum_test(index_list, "Spearman", "TradingBot/Data/DownMarket/",  alpha_val = a, num_terms=20, interval=250)
#     print("Index: " + index + " --- Sharpe: " + str(round(sh,2)) + " --- CAG: " + str(round(cag,2)))
#     avg_cag += cag
#     avg_sh += sh

# print("Average: " + str(avg_cag/7) + "  " + str(avg_sh/7))

# a = 0.85
# v, sh, cag, e = mlink_momentum_test(index_list, "Spearman", "TradingBot/Data/",  alpha_val = a, num_terms=1, interval=250)
# print("Combined --- Sharpe: " + str(round(sh,2)) + " --- CAG: " + str(round(cag,2)))

# print("Average: " + str(avg_cag/7) + "  " + str(avg_sh/7))

# for a in [.1, .15, .2, .25, .3, .5, .85]:

#     for n_terms in [1,5,10,15,20]:
#         v, sh, cag, e = mlink_momentum_test(index_list, "Spearman", "TradingBot/Data/",  alpha_val = a, num_terms=n_terms, interval=250)
#         print("Alpha: " + str(a) + " --- Terms: " + str(n_terms) + " --- Sharpe: " + str(round(sh,2)) + " --- CAG: " + str(round(cag,2)))


#     print("\n----\n")

# Testing with new difference works well.
# Quant Connect testing with absdif working well so far.