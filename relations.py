from BacktestMethods import ResearchTest
from Algorithms import LinkMomentum, SectorMomentum, MultipleLinkMomentum

import pandas as pd
import numpy as np

def link_momentum_test(d,corr,path):
    # for interval lengths (in days)
    for interval in [120]:
        plot_pre=d[:-1] + str(interval)
        print(d[:-1])
        # print(interval)

        # open the data for the correlation matrix
        with open(path+d+corr+"/"+str(interval)+"Day.pkl", 'rb') as handle:
            arr=pd.read_pickle(handle)
            op=pd.read_pickle(path+d+'Open.pkl') # load Open Data
            df=pd.read_pickle(path+d+'Close.pkl') # load Close Data

            # op.to_csv("Open.csv")
            # df.to_csv("Close.csv")

            # get the first, last keys (dates) and the indices for corresponding keys
            first_key=list(arr.keys())[0]
            last_key=list(arr.keys())[-1]
            start_index=list(str(d.date()) for d in df.index).index(first_key)
            end_index=list(str(d.date()) for d in df.index).index(last_key)

            ticks=list(i for i in op.columns[:int(len(op.columns)/2)]) # stock tickers to look at

            start_date=list(d for d in df.index)[start_index] # first date of trading
            end_date=list(d for d in df.index)[end_index] # last date of trading
            print(len(ticks))


        lm = LinkMomentum(arr,ticks,k2=1,method="dif") # initialize Algorithm
        bt = ResearchTest(lm,df,op,tickers=ticks,start_date=start_date,end_date=end_date,dates=list(d.date() for d in df.index)[start_index:end_index], above_avg_value=True, graph_signal=False) # initialize backtest
        bt.backtest() # backtest

        # print("Sector Performance:", bt.sector_performance[0])
        # print("Algorithm Signals: ", bt.alg_average_signals[0])
        # # print("Average Eigenvalue: ", bt.leig)
        # print("Normalized Average Eigenvalue: ", bt.leig/len(ticks))

        # -plot the algorithms-
        bt.nplot_all(plot_pre)
        # bt.plot_signal(plot_pre+"Signal")
        # bt.plot_vol(plot_pre+"Vol")


        # plot the algorithms
        # bt.plot_prob(plot_pre)
        # bt.plot_all(plot_pre)
        # bt.nplot_all(plot_pre)

def multi_momentum_test(dir_list,bs=False,av=False):
    op = []
    df = []
    ticks = []

    path ="TradingBot/Data/"
    corr="Spearman"
    first_test = True

    p = ""
    if bs:
        p = "bs"
    if av:
        p = "av"

    # for interval lengths (in days)
    for interval in [30,60,120,250]:
        print("Interval: ", interval)
        print("Sectors: ", [sect[:-1] for sect in dir_list])
        plot_pre=p+str(interval)
        algs = []
        arrs = []

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

                if first_test:
                    print(start_date)
                    print(end_date)
                    first_test = False
                    
                # Short Term fix for offset time periods inter-sector data
                dates = list(x.date() for x in close_data.index)[start_index+20:-20]

                ticks.append(t)
                op.append(open_data)
                df.append(close_data)
                arrs.append(arr)
                algs.append(LinkMomentum(arr,t,k2=1,method="dif"))

                print(t)

        bt = ResearchTest(algs,df,op,tickers=ticks,start_date=start_date,end_date=end_date, dates=dates, multi_algs=True, best_signal=bs, above_avg_value=av) # initialize backtest
        bt.backtest() # backtest

        # plot the algorithms
        # bt.plot_prob(plot_pre)
        bt.nplot_all(plot_pre)

        print("Buy Count: ", bt.buy_count)
        print("Sell Count: ", bt.sell_count)


# for s in ["Finance/", "Electric/", "Tech/", "Energy/", "Health/", "Indust/", "REstate/"]:
    # link_momentum_test(s,"Spearman", "TradingBot/Data/DownMarket/")

link_momentum_test("Finance/","Spearman", "TradingBot/Data/DownMarket/")

# multi_momentum_test(["Finance/", "Electric/", "Tech/", "ConsD/", "Energy/", "Health/", "Indust/", "REstate/"])
# print()
# print("-------")
# print()

# multi_momentum_test(["Finance/", "Electric/", "Tech/", "ConsD/", "Energy/", "Health/", "Indust/", "REstate/"],bs=True)
# print()
# print("-------")
# print()
# multi_momentum_test(["Finance/", "Electric/", "Tech/", "ConsD/", "Energy/", "Health/", "Indust/", "REstate/"],av=True)