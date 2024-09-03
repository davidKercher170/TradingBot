from audioop import mul
from cProfile import label
import datetime
import datetime as dt
import profile
from signal import signal
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

import tkinter
import tkinter as tk
import tkinter.scrolledtext as scrolledtext

from pandas.tseries.offsets import BDay
from tkinter import ttk
from PIL import ImageTk, Image

from scipy.linalg import lstsq

class Backtest():
    '''
    Backtests algorithm. Can return varying plots, performance scores, and computations.
    '''
    
    def __init__(self,alg,dff,opp,start_date=(datetime.date.today()-BDay(300)).date(),end_date=dt.date.today(),dates=[],initial_balance=100000,
                 tickers=[],meas_stock=[],adv_plots=True, multi_algs = False, signal_thresholds = False, leverage=1, sector_momentum=[], P=1, monthly_returns=False):
        """
        Backtest Initialization. To Avoid Confusion will ONLY buy or sell at market open (and at open price).
        Initial start_date is 300 business days prior, initial end date is today's date.
        :param alg: Algorithm to run.
        :param dff: Dataframe containing close price data.
        :param opp: Dataframe containing open price data.
        :param start_date: Date to begin trading.
        :param end_date: Date to end trading.
        :param dates: List of dates to trade on.
        :param initial_balance: Initial balance to trade with.
        :param tickers: List of stock tickers to trade.
        :param meas_stock: Stock to measure against (Baseline Stock).
        :param adv_plots: When set to true will make use of advance plots (volume, volatility,...). Otherwise will simply return overall return graph.
        """

        self.monthly_returns=monthly_returns

        self.algorithm = alg # algorithm is the given algorithm object to test
        self.tickers=tickers # stock symbols
        self.num_stocks = len(tickers)
        self.dates = dates 
        self.mes = meas_stock
        self.signal_vals = []
        self.leverage = leverage
        self.P = P # number of days between trades

        if len(sector_momentum) > 0:
            self.sector_momentum = sector_momentum
            self.num_stocks -= 1

        if signal_thresholds != False:
            self.buy_thresholds = signal_thresholds[0]
            self.sell_thresholds = signal_thresholds[1]


        self.multi_algs = multi_algs
        self.holdings = {} 
        if self.multi_algs:
            self.num_algs = len(alg)
            self.list_holdings = [{} for j in range(self.num_algs)]
            for i in range(self.num_algs):
                for j in range(len(self.tickers[i])): self.list_holdings[i][j] = (0,0,0)

        else:
            for i in range(self.num_stocks): self.holdings[i] = (0,0,0)
    
            
        self.start_date = start_date # start date for test
        self.end_date = end_date # end date for test
        self.df = pd.DataFrame() # Dataframe to store data
        self.num_bdays = 0 # number of business days to trade
        self.adv_plots=adv_plots
        #key : (percentage of portfolio, price, and number of shares)
        
        self.initial_balance = initial_balance # starting balance
        self.balance = initial_balance # current balance
        self.tot_balance = self.balance # total balance (including stocks)
        
        self.time_period = 60 # time period for correlation 
        
        self.win = 0 # number of trades that resulted in gain
        self.win_total = 0 # total gain over all wins
        self.loss = 0 # number of trades that resulted in loss
        self.loss_total = 0 # total loss over all losses
        self.trades = 0 # total number of trades
        self.current_iteration = 0 # current iteration (iterate from 0 to number of business days)
        self.log = "" # log that will be output in GUI, keeps track of trades/balance/gain/...
        
        self.riskfree = 1.015 # risk free rate is 3%, used to calculate sharpe
        
        self.trade_dates = [] # dates that a trade was made on
        self.trade_balances = [] # balance after each trade
        self.interest_vals = [] # total return interest after each trade
        self.sinterest_vals = [100] # interest vals for sells
        self.binterest_vals = [100] # interest vals for buys
        self.real_start_date = self.start_date
        self.ind_trade_dates = [self.real_start_date]
        self.vol_prices = []
        
        
        self.date = ''
        
        self.df = dff
        self.open = opp
        self.num_bdays = len(self.dates)

            
        self.bvolume = {}
        self.svolume = {}
        self.bvol_count = 0
        self.svol_count = 0

        for i in range(self.num_stocks): 
            self.bvolume[i] = 0
            self.svolume[i] = 0

    def monthly_stats(self):
        m_ret = [(self.interest_vals[k]/self.interest_vals[k-22]-1)*100 for k in range(22, len(self.interest_vals), 22)]

        # Compute %MktShare
        mkt_share = 0

        # Compute Cap
        cap = 0

        # Compute B/M
        bm = 0

        # Compute CAPM
        capm = 0

        # Compute FF-3
        ff = 0

        return np.average(m_ret), np.std(m_ret)
        
    def calc_vol(self):

        """
        Calculate approximate volatility
        """
        if len(self.interest_vals) <= 260:
            return np.std([i-100 for i in self.interest_vals])

        r = int(len(self.interest_vals)/21)
        p = [(self.interest_vals[(j+1)*21]-self.interest_vals[j*21]) / self.interest_vals[j*21] * 100 for j in range(r)]
        return np.std(p)

    def calc_tot_bal(self):
        """
        Calculate Total Balance : Free Balance + Unrealized Capital
        """
        for k in self.holdings.keys():
            if self.holdings[k][2] != 0: self.tot_balance += self.holdings[k][2]*self.open.iloc[:,k][self.current_iteration]
        return self.tot_balance
        
    def backtest(self):
        """
        Overarching Backtest Method.
        """

        if self.multi_algs:
            self.avg_bsignal = [0 for i in range(self.num_algs)]
            self.avg_ssignal = [0 for i in range(self.num_algs)]

            for itera in range(self.time_period,self.num_bdays-2,self.P):
                self.sgain=0
                self.bgain=0
                self.current_iteration=itera
                self.date = str(self.dates[itera])

                self.num_buys = 0
                self.num_sells = 0
                buys = []
                sells = []

                for alg_ind in range(self.num_algs):
                    buy, sell, c = self.algorithm[alg_ind].alpha(self.date,self.open[alg_ind].loc[[self.date]],itera)
                    vb = c[0] / 100
                    vs = c[1] / 100

                    # sell = []
                    if vb < 0:
                        buy = []
                        sell = []
                    else:
                        self.num_buys += 1 * len(buy)
                        self.num_sells += 1 * len(sell)

                    # if vs > 0:
                    #     sell = []
                    # else:
                    #     self.num_sells += 1



                    # self.avg_bsignal[alg_ind] += vb  / self.num_bdays
                    self.avg_ssignal[alg_ind] += vs  / self.num_bdays
                    buys.append(buy)
                    sells.append(sell)

                for alg_ind in range(self.num_algs):
                    self.liquidate_all(buys[alg_ind],sells[alg_ind],alg_ind)

                self.pre_trade_bal = self.balance

                for alg_ind in range(self.num_algs):
                    self.set_holdings(buys[alg_ind],sells[alg_ind],alg_ind)

                self.trade_dates.append(self.dates[itera])  # add current date (for plots)
                self.trade_balances.append(self.tot_balance) # add current total balance (for plot)
                self.interest_vals.append(self.tot_balance/self.initial_balance*100) # add current total return interest (for plots)

                self.ind_trade_dates.append(self.date)
                self.sinterest_vals.append((self.sinterest_vals[-1])+self.sgain/self.initial_balance*100) # interest vals for sells
                self.binterest_vals.append((self.binterest_vals[-1])+self.bgain/self.initial_balance*100) # interest vals for buys
                self.signal_vals.append(self.sgain/self.initial_balance)

        else:
            for i in range(self.time_period,self.num_bdays-2,1):
                self.current_iteration=i
                self.date = str(self.dates[i])
                # original method for central method
                buy, sell, c = self.algorithm.alpha(self.date,self.open.loc[[self.date]],self.current_iteration)
                self.set_holdings(buy,sell)
             
    def liquidate_all(self,buy,sell,index_val=0):

        if self.multi_algs:

            # liquidate stocks that were in holdings but no longer have signals
            for h in self.list_holdings[index_val].keys():

                # if stock has shares (is in portofolio)
                if self.list_holdings[index_val][h][2]!=0:

                    # if stock is not in buy or sell signals
                    if h not in [b[0] for b in buy] and h not in [b[0] for b in sell]:
                        self.liquidate(h,int(self.list_holdings[index_val][h][2]/abs(self.list_holdings[index_val][h][2])), 1, index_val)
                        
                    # a stock with short position now has a buy signal
                    elif h in [b[0] for b in buy] and self.list_holdings[index_val][h][2] < 0:
                        self.liquidate(h,-1,1,index_val)
                        
                    # a stock with long position now has a sell signal
                    elif h in [b[0] for b in sell] and self.list_holdings[index_val][h][2] > 0:
                        self.liquidate(h,1,1,index_val)

                    # Liquidate stocks that no longer have sell signal
                    elif h in [b[0] for b in sell]:
                        self.liquidate(h,-1,1,index_val)

                    # Liquidate stocks that no longer have buy signal
                    elif h in [b[0] for b in buy]:
                        self.liquidate(h,1,1,index_val)

    def set_holdings(self,buy,sell,index_val=0):
        """
        Sets portfolio holdings.
        :param buy: list of stock tickers to buy
        :param long: list of stock tickers to sell
        """

        if len(buy) == 0 or len(sell) == 0: div = 1
        else: div = 2
        
        if self.multi_algs:
                    
            # used pre trade bal to calculate percentage of each stock to buy (as percentage of original balance vs balance after each trade)
            # which would screw up number of stocks to buy

            # sell represents list of tuples (symbol, percentage of funds to allocate)
            for b in sell:
                open_price = self.open[index_val][self.tickers[index_val][b[0]]].loc[self.date]# today's open price
                self.svol_count += 1
                
                # if stock to buy/sell is already in holdings
                if self.list_holdings[index_val][b[0]][2] != 0:
                    self.liquidate(b[0],-1,0,index_val)
                    self.log += self.date + ". Stock To Short Already In Holdings. " + str(self.tickers[index_val][b[0]]) + "\n"

                # if the stock is not in holdings and we have the available funds to buy stock
                elif self.pre_trade_bal/self.num_sells/div > open_price:
                    shares = int(self.pre_trade_bal/self.num_sells/open_price/div) # actual number of shares to buy (must be integer) - will always round down
                    perc = (open_price*shares)/self.pre_trade_bal# actual percentage of balance allocated
                    self.balance = self.balance-(open_price*shares) #  new balance
                    self.list_holdings[index_val][b[0]] = [perc, open_price, -1*shares] # set holdings with percentage of portfolio, price, and number of shares
                    
                    # add buy info to log
                    self.log += self.date+". B:"+str(self.tickers[index_val][b[0]]) + " [Pr:" + str(round(open_price,2))+"]"+"[#"+str(shares)+"][Bal: "+str(round(self.balance,2))\
                    +"][TBal: "+str(round(self.tot_balance,2))+"][-1]\n"
                
                # Otherwise no funds available and stock is not already in holdings
                else:
                    self.log += self.date + ". None. " + str(self.tickers[index_val][b[0]]) + "\n"
            
            # b represents list of tuples (symbol, percentage of funds to allocate)
            for b in buy:

                open_price = self.open[index_val][self.tickers[index_val][b[0]]].loc[self.date] # today's open price
                self.bvol_count += 1
                
                # if stock to buy/sell is already in holdings
                if self.list_holdings[index_val][b[0]][2] != 0:
                    self.liquidate(b[0],1,0,index_val)
                    self.log += self.date +". Stock To Buy Already In Holdings. "+str(self.tickers[index_val][b[0]]) + "\n"

                # if the stock is not in holdings and we have the available funds to buy stock
                elif self.pre_trade_bal/self.num_buys/div > open_price:
                    shares = int(self.pre_trade_bal/self.num_buys/open_price/div) # actual number of shares to buy (must be integer) - will always round down
                    perc = (open_price*shares)/self.pre_trade_bal # actual percentage of balance allocated
                    self.balance = self.balance-(open_price*shares) #  new balance
                    self.list_holdings[index_val][b[0]] = [perc, open_price, shares] # set holdings with percentage of portfolio, price, and number of shares
                    
                    # add buy info to log
                    self.log += self.date+". B:"+str(self.tickers[index_val][b[0]]) + " [Pr:" + str(round(open_price,2))+"]"+"[#"+str(shares)+"][Bal: "+str(round(self.balance,2))\
                    +"][TBal: "+str(round(self.tot_balance,2))+"][1]\n"
                
                # Otherwise no funds available and stock is not already in holdings
                else:
                    self.log += self.date + ". None. " + str(self.tickers[index_val][b[0]]) + "\n"

        else:
            for h in self.holdings.keys():

                    # if stock has shares (is in portofolio)
                    if self.holdings[h][2]!=0:

                        # if stock is not in buy or sell signals
                        if h not in [b[0] for b in buy] and h not in [b[0] for b in sell]:
                            self.liquidate(h,int(self.holdings[h][2]/abs(self.holdings[h][2])))
                            
                        # a stock with short position now has a buy signal
                        elif h in [b[0] for b in buy] and self.holdings[h][2] < 0:
                            self.liquidate(h,-1)
                            
                        # a stock with long position now has a sell signal
                        elif h in [b[0] for b in sell] and self.holdings[h][2] > 0:
                            self.liquidate(h,1)

                        # Liquidate stocks that no longer have sell signal
                        elif h in [b[0] for b in sell]:
                            self.liquidate(h,-1)

                        # Liquidate stocks that no longer have buy signal
                        elif h in [b[0] for b in buy]:
                            self.liquidate(h,1)
                        # liquidate stocks that were in holdings but no longer have signals

            # used pre trade bal to calculate percentage of each stock to buy (as percentage of original balance vs balance after each trade)
            # which would screw up number of stocks to buy
            pre_trade_bal = self.balance
        
            # sell represents list of tuples (symbol, percentage of funds to allocate)
            for b in sell:
                open_price = self.open[self.tickers[b[0]]].loc[self.date]# today's open price
                self.svolume[b[0]] += 1 # 'volume' just indicates how many times there was a signal to buy this stock
                self.svol_count += 1
                
                # if stock to buy/sell is already in holdings
                if self.holdings[b[0]][2] != 0:
                    self.liquidate(b,-1,0)
                    self.log += self.date + ". Stock To Short Already In Holdings. " + str(self.tickers[b[0]]) + "\n"

                # if the stock is not in holdings and we have the available funds to buy stock
                elif self.balance > open_price:
                    shares = int(b[1]*pre_trade_bal/open_price) # actual number of shares to buy (must be integer) - will always round down
                    perc = (open_price*shares)/pre_trade_bal# actual percentage of balance allocated
                    self.balance = self.balance-(open_price*shares) #  new balance
                    self.holdings[b[0]] = [perc, open_price, -1*shares] # set holdings with percentage of portfolio, price, and number of shares
                    
                    # add buy info to log
                    self.log += self.date+". B:"+str(self.tickers[b[0]]) + " [Pr:" + str(round(open_price,2))+"]"+"[#"+str(shares)+"][Bal: "+str(round(self.balance,2))\
                    +"][TBal: "+str(round(self.tot_balance,2))+"][-1]\n"
                
                # Otherwise no funds available and stock is not already in holdings
                else:
                    self.log += self.date + ". None. " + str(self.tickers[b[0]]) + "\n"
            
            # b represents list of tuples (symbol, percentage of funds to allocate)
            for b in buy:
                open_price = self.open[self.tickers[b[0]]].loc[self.date] # today's open price
                self.bvolume[b[0]] += 1 # 'volume' just indicates how many times there was a signal to buy this stock
                self.bvol_count += 1
                
                # if stock to buy/sell is already in holdings
                if self.holdings[b[0]][2] != 0:
                    self.liquidate(b,1,0)
                    self.log += self.date +". Stock To Buy Already In Holdings. "+str(self.tickers[b[0]]) + "\n"

                # if the stock is not in holdings and we have the available funds to buy stock
                elif self.balance > open_price:
                    shares = int(b[1]*pre_trade_bal/open_price) # actual number of shares to buy (must be integer) - will always round down
                    perc = (open_price*shares)/pre_trade_bal# actual percentage of balance allocated
                    self.balance = self.balance-(open_price*shares) #  new balance
                    self.holdings[b[0]] = [perc, open_price, shares] # set holdings with percentage of portfolio, price, and number of shares
                    
                    # add buy info to log
                    self.log += self.date+". B:"+str(self.tickers[b[0]]) + " [Pr:" + str(round(open_price,2))+"]"+"[#"+str(shares)+"][Bal: "+str(round(self.balance,2))\
                    +"][TBal: "+str(round(self.tot_balance,2))+"][1]\n"
                
                # Otherwise no funds available and stock is not already in holdings
                else:
                    self.log += self.date + ". None. " + str(self.tickers[b[0]]) + "\n"
                
            # Update Information for plots
            
            self.vol_prices.append(self.tot_balance/self.initial_balance*100)

            # if self.current_iteration % int(self.num_bdays/10) == 0 or self.date==self.end_date:
            self.trade_dates.append(self.dates[self.current_iteration])  # add current date (for plots)
            self.trade_balances.append(self.tot_balance) # add current total balance (for plot)
            self.interest_vals.append(self.tot_balance/self.initial_balance*100) # add current total return interest (for plots)

            self.ind_trade_dates.append(self.date)
            self.sinterest_vals.append((self.sinterest_vals[-1])+self.sgain/self.initial_balance*100) # interest vals for sells
            self.binterest_vals.append((self.binterest_vals[-1])+self.bgain/self.initial_balance*100) # interest vals for buys
            
            # add current total return interest for each standard stock(for plots/to measure against alg)
            if len(self.standard) > 0:
                for i in range(len(self.standard)):
                    self.interest_stand[i].append(self.open[self.standard[i]][self.current_iteration]/self.standard_initial[i]*100)
            
    def liquidate(self, s, long=1, v=1, index_val=0):
        """
        Liquidates given position.
        :param s: stock ticker
        :param long: 
        :param v:
        """

        if self.multi_algs:
            # open price for the day
            open_price = round(self.open[index_val][self.tickers[index_val][s]].loc[self.date],2) # current open price
            num_shares = self.list_holdings[index_val][s][2] # number of shares held
            bought_price = self.list_holdings[index_val][s][1] # price bought at (open since all transactions are at open in test)
            
            # add iteration to to log
            self.log += self.date +". "
            
            # gain (or loss) from trade
            gain = (abs(open_price)-abs(bought_price))*num_shares*self.leverage
            self.tot_balance += gain # calculate new total balance

            if long==1:
                self.bgain += gain # interest vals for buys
            else:
                self.sgain += gain # interest vals for sells
            
            # update balances but keep in holdings
            if v == 0:
                return
            
            if gain > 0:
                self.win +=1 # add 1 to win
                self.log += "+" # add to log to indicate trade was positive
                self.win_total += gain/bought_price/num_shares # add percentage value of win
                
            else:
                self.loss += 1 # add 1 to loss
                self.log += "-" # add to log to indicate trade was positive
                self.loss_total += gain/bought_price/num_shares # add percentage value of loss


            self.trades += 1 # total number of trades - (does not account for unliquidated trades)
            # self.balance += long*num_shares*bought_price # update balance incorrect
            self.balance += long*num_shares*bought_price + gain # actually update balance
            self.list_holdings[index_val][s] = (0,0,0) # Reset Holdings Value

            # Log Sale
            self.log += "S:"+str(self.tickers[index_val][s])+" [Pr:"+str(open_price)+"][#"+str(num_shares)+"][Bal:" + str(round(self.balance,2))\
            +"][TBal:"+str(round(self.tot_balance,2))+"][+/-:"+str(round(gain,2))+"]["+str(long)+"]\n"

        else:
            # open price for the day
            open_price = round(self.open[self.tickers[s]].loc[self.date],2) # current open price
            num_shares = self.holdings[s][2] # number of shares held
            bought_price = self.holdings[s][1] # price bought at (open since all transactions are at open in test)
            
            # add iteration to to log
            self.log += self.date +". "
            
            # gain (or loss) from trade
            gain = (abs(open_price)-abs(bought_price))*num_shares
            self.tot_balance += gain # calculate new total balance

            if long==1:
                self.bgain += gain # interest vals for buys
            else:
                self.sgain += gain # interest vals for sells
            
            # update balances but keep in holdings
            if v == 0:
                return
            
            if gain >= 0:
                self.win +=1 # add 1 to win
                self.log+="+" # add to log to indicate trade was positive
                self.win_total += gain/bought_price # add percentage value of win
                
            else:
                self.loss +=1 # add 1 to loss
                self.log += "-" # add to log to indicate trade was positive
                self.loss_total += gain/bought_price # add percentage value of loss


            self.trades += 1 # total number of trades - (does not account for unliquidated trades)
            self.balance += long*num_shares*bought_price + gain # update balance
            self.holdings[s] = (0,0,0) # Reset Holdings Value

            # Log Sale
            self.log += "S:"+str(self.tickers[s])+" [Pr:"+str(open_price)+"][#"+str(num_shares)+"][Bal:" + str(round(self.balance,2))\
            +"][TBal:"+str(round(self.tot_balance,2))+"][+/-:"+str(round(gain,2))+"]["+str(long)+"]\n"

    def calc_prob(self):
        return self.win/self.trades, self.loss/self.trades

    def eval(self):
        return (self.win_total+self.loss_total)/self.trades*100

    def comp_ret(self):
        """
        Computes the total Return over window.
        :return: total return
        """
        return (self.interest_vals[-1]-100)
       
    def comp_annual(self):
        """
        Computes the average compounding annual return.
        :return: average compounding annual return
        """
        if self.comp_ret() < 0:
            return -1*abs(((self.comp_ret()+100)/100)**(260/self.num_bdays)-1)*100
        return (((self.comp_ret()+100)/100)**(260/self.num_bdays)-1)*100
    
    def sharpe(self):
        """
        Computes (approximately) the sharpe ratio over trading period
        :return: returns sharpe ratio
        """
        return (self.comp_annual()-self.riskfree*100+100)/self.calc_vol()

    def alt_sharpe(self):
        """
        Computes (approximately) the sharpe ratio over trading period by 
        comuting the return variance from the average return in each incremental period, summing the squares, and dividing by
        the number of time periods, then finally taking the square root of the quotient.
        :return: returns sharpe ratio
        """

        i_vals = self.mes["^GSPC"]

        if len(i_vals) <= 260:
            v = np.std([i-100 for i in i_vals])
        else:
            r = int(len(self.interest_vals)/21)
            p = [(self.interest_vals[(j+1)*21]-self.interest_vals[j*21]) / self.interest_vals[j*21] * 100 for j in range(r)]
            v = np.std(p)

        # print(i_vals[-1]/i_vals[0])
        # print(v)

        if i_vals[-1]/i_vals[0] < 0:
            return 0
        c = (((i_vals[-1]/i_vals[0]))**(260/self.num_bdays)-1)*100

        # print("CAG:", c)
        # print((c-self.riskfree*100+100)/v)
        return (c-self.riskfree*100+100)/v
    
    def group_performace(self, stock_average=False):
        stocks_tot = []

        if stock_average:
            for d in self.trade_dates:
                tot = 0
                for tick in self.tickers[0]:
                    if self.open[0][tick].loc[str(self.dates[self.time_period])] != 0:
                        tot += self.open[0][tick].loc[str(d)]/self.open[0][tick].loc[str(self.dates[self.time_period])]
                stocks_tot.append(tot/ len(self.tickers[0])*100)
            # print(stocks_tot[-1] - stocks_tot[0])
            # print((((stocks_tot[-1] - stocks_tot[0]+100)/100)**(260/self.num_bdays)-1)*100)
            return stocks_tot

        if self.multi_algs:
            g = [100]+[float(self.mes["^GSPC"][i])/float(self.mes["^GSPC"][0])*100 for i in range(self.time_period+1,self.num_bdays-2)]
            # print(g[-1] - g[0])
            # print((((g[-1] - g[0]+100)/100)**(260/self.num_bdays)-1)*100)
            return g

        for d in self.trade_dates:
            tot = 0
            for tick in self.tickers:
                if self.open[tick].loc[str(self.dates[self.time_period])] != 0:
                    tot += self.open[tick].loc[str(d)]/self.open[tick].loc[str(self.dates[self.time_period])]
            stocks_tot.append(tot/ len(self.tickers)*100)
        return stocks_tot

    def plot(self,pref):
        """
        Plots overall algorithm performance over trading window.
        :return: plt plot of performance
        """ 
        
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(12)           

        mpl.rcParams['font.size'] = 25
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=240))
        # plt.xticks(rotation=90)
        
        plt.plot(self.trade_dates, self.interest_vals, linewidth=2.2, color='green', label="Algorithm")
        plt.plot(self.trade_dates, self.sinterest_vals[1:], linewidth=2.2, color='red', label="Sell")
        plt.plot(self.trade_dates, self.binterest_vals[1:], linewidth=2.2, color='blue', label="Buy")
        plt.plot(self.trade_dates, self.group_performace(), linewidth=2.2, label="SP500")


        if self.num_algs == 1:
            plt.plot(self.trade_dates, self.group_performace(stock_average=True), linewidth=2.2, label="Sector")

        plt.legend(fontsize=30)
        plt.savefig(pref+"Ret")
        plt.savefig(pref+"Ret",format='svg')

        self.alt_sharpe()
        return pref+"Ret"
    
    def indplot(self):
        """
        Plots performance of long and short positions independently over trading window.
        :return: plt plot of long/short performance
        """ 
        days = mdates.drange(self.ind_trade_dates[0], self.ind_trade_dates[-1],dt.timedelta(days=1))
        
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(10)              

        mpl.rcParams['font.size'] = 25
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=int(self.num_bdays/5)))
        
        plt.plot(self.ind_trade_dates, self.binterest_vals, linewidth=4, color='blue', label="Long Position")
        plt.plot(self.ind_trade_dates, self.sinterest_vals, linewidth=4, color='red', label="Short Position")
        
        # plot graphs for given measuring stocks (stocks to measure against algorithm)
        for i in range(len(self.standard)):
            plt.plot(self.trade_dates, self.interest_stand[i], linewidth=4,label=self.standard[i])
        
        plt.legend()
        plt.savefig('lsret_plot')
        return "lsret_plot.png"
        
    def plot_bvol(self):
        """
        Plots volatility of bought stocks over trading window.
        :return: Plot of bought volume
        """ 
        fig = plt.figure()
        
        fig.set_figwidth(20)
        fig.set_figheight(10)         
        
        ax = fig.add_subplot(111)
        stocks = list(self.bvolume.keys())
        threshold = np.average(list(self.bvolume.values()))*0.5
        x,y  = [],[]
        
        for i in stocks:
            v = self.bvolume[i]
            if v >= threshold:
                x.append(self.tickers[i])
                y.append(v/self.bvol_count)

        ax.bar(x,y)
        plt.xticks(rotation='vertical')
        plt.ylabel("% Trades")
        
        plt.savefig('vol')
        return "vol.png"
    
    def plot_svol(self):

        """
        Plots volume of sold stocks over trading window.
        :return: Plot of shorted volume
        """ 
        fig = plt.figure()
        
        fig.set_figwidth(20)
        fig.set_figheight(10)         
        
        ax = fig.add_subplot(111)
        stocks = list(self.svolume.keys())
        threshold = np.average(list(self.svolume.values()))*0.5
        x,y  = [],[]
        
        for i in stocks:
            v = self.svolume[i]
            if v >= threshold:
                x.append(self.tickers[i])
                y.append(v/self.svol_count)

        ax.bar(x,y)
        plt.xticks(rotation='vertical')
        plt.ylabel("% Trades")
        
        plt.savefig('vol')
        return "vol.png"


class ResearchTest():
    '''
    Backtests algorithm using research methods. Can return varying plots, performance scores, and computations.
    '''
    
    def __init__(self,alg,dff,opp,start_date=(datetime.date.today()-BDay(300)).date(),end_date=dt.date.today(),dates=[],tickers=[], multi_algs=False, best_signal=False, above_avg_value=False,
     graph_signal=False,volatility=True):
        """
        Backtest Initialization. To Avoid Confusion will ONLY buy or sell at market open (and at open price).
        Initial start_date is 300 business days prior, initial end date is today's date.
        :param alg: Algorithm to run.
        :param dff: Dataframe containing close price data.
        :param opp: Dataframe containing open price data.
        :param start_date: Date to begin trading.
        :param end_date: Date to end trading.
        :param dates: List of dates to trade on.
        :param initial_balance: Initial balance to trade with.
        :param tickers: List of stock tickers to trade.
        :param meas_stock: Stock to measure against (Baseline Stock).
        :param adv_plots: When set to true will make use of advance plots (volume, volatility,...). Otherwise will simply return overall return graph.
        """
        self.algorithm = alg # algorithm is the given algorithm object to test
        self.tickers=tickers # stock symbols
        self.num_stocks = len(tickers)
        self.dates = dates
        self.multi_algs = multi_algs
        self.average_value = above_avg_value
        self.vol = True
        
        
        self.graph_signal = graph_signal # if true will graph Signal-Profit
        self.buy_signals = []
        self.sell_signals = []
        self.week_values = []
        self.month_values = []
        self.three_month_values = []

        self.best_signal = best_signal # if true will only trade highest signal

        
        self.start_date = start_date # start date for test
        self.end_date = end_date # end date for test
        self.df = pd.DataFrame() # Dataframe to store data

        self.buys = {} # key : [[iteration, buy_value, day_value, week_value, month_value, year_value]] (list of signals for each stock)
        self.sells = {} # key : [[iteration, buy_value, day_value, week_value, month_value, year_value]] (list of signals for each stock)
        self.comb = [] # key : [[iteration, buy_value, day_value, week_value, month_value, year_value]] (list of signals for each stock)

        self.num_bdays = len(self.dates) # number of business days
        self.time_period = 500

        self.df = dff
        self.open = opp
        self.num_bdays = len(self.dates)

        self.average_buy = 0
        self.average_sell = 0
        self.leig = 0

        if multi_algs == True:
            self.num_algs = len(self.algorithm)
        else:
            self.num_algs = 1

        self.alg_average_signals = [[0,0] for i in range(self.num_algs)]
        self.sector_performance = [0 for i in range(self.num_algs)]
        self.buy_count = [0 for i in range(self.num_algs)]
        self.sell_count = [0 for i in range(self.num_algs)]

    def backtest(self):
        """
        Overarching Backtest Method.
        """
        for i in range(self.time_period,self.num_bdays-2):
        # for i in range(self.time_period,self.time_period+5):
            self.current_iteration=i
            self.date = str(self.dates[i])

            # multiple algorithms
            if self.multi_algs:

                # trade best signal
                if self.best_signal:
                    trades = [self.algorithm[j].alpha(self.date,self.open[j].loc[[self.date]],self.current_iteration) for j in range(self.num_algs)]
                    b = np.argmax([trades[m][2][0] for m in range(self.num_algs)])
                    s = np.argmin([trades[n][2][1] for n in range(self.num_algs)])
                    self.add_trades(trades[b][0],"NA",b)
                    self.add_trades("NA",trades[s][1],s)
                    self.buy_count[b] += 1
                    self.sell_count[s] += 1

                # trade all or trade above average signals
                else:
                    for j in range(self.num_algs):
                        buy, sell, c = self.algorithm[j].alpha(self.date,self.open[j].loc[[self.date]],self.current_iteration)

                        # get average values
                        avg1 = c[0]/len(self.tickers[j])
                        avg2 = c[1]/len(self.tickers[j])
                        self.alg_average_signals[j][0] += avg1 / self.num_bdays
                        self.alg_average_signals[j][1] += avg2 / self.num_bdays

                        # Used to test performance based on signal strength
                        if self.average_value == True:
                            if avg1 < 0.1:
                                buy = "NA"

                            if avg2 > -0.07:
                                sell = "NA"

                        self.add_trades(buy,sell,j)

            # if a single algorithm
            else:
                buy, sell, c = self.algorithm.alpha(self.date,self.open.loc[[self.date]],self.current_iteration)
                self.leig += self.algorithm.leig / self.num_bdays

                if self.graph_signal:
                    if c[0] > 10 or c[0] < -10:
                        continue
                    self.buy_signals.append(c[0] / self.num_stocks)
                    self.sell_signals.append(c[1] / self.num_stocks)
                    avg1 = c[0]
                    avg2 = c[1]

                # Used to test performance based on signal strength
                if self.average_value == True:
                    avg1 = c[0] / self.num_stocks
                    avg2 = c[1] / self.num_stocks

                    self.alg_average_signals[0][0] += avg1 / self.num_bdays
                    self.alg_average_signals[0][1] += avg2 / self.num_bdays

                    # Experimental Values based on Signal Plots
                    if avg1 < 0.1:
                        buy = "NA"

                    if avg2 > -0.05:
                        sell = "NA"

                if len(buy) != 0:
                    self.add_trades(buy,sell)

    def add_trade(self, x, index=0):
        """
        Adds stock x to the list of trades
        :param x: stock ticker x
        :param index: index to trade
        """

        if self.multi_algs:
            curr_value =  self.open[index][self.tickers[index][x]].loc[str(self.dates[self.current_iteration])]
            day_value =  self.open[index][self.tickers[index][x]].loc[str(self.dates[self.current_iteration+1])]

            try:
                week_value =  self.open[index][self.tickers[index][x]].loc[str(self.dates[self.current_iteration+5])]
            except Exception:
                week_value = 0

            try:
                month_value =  self.open[index][self.tickers[index][x]].loc[str(self.dates[self.current_iteration+25])]
            except Exception:
                month_value = 0

            try:
                three_month_value =  self.open[index][self.tickers[index][x]].loc[str(self.dates[self.current_iteration+75])]
            except Exception:
                three_month_value = 0

            try:
                six_month_value =  self.open[index][self.tickers[index][x]].loc[str(self.dates[self.current_iteration+150])]
            except Exception:
                six_month_value = 0

            try:
                year_value =  self.open[index][self.tickers[index][x]].loc[str(self.dates[self.current_iteration+300])]
            except Exception:
                year_value = 0

            try:
                two_year_value =  self.open[index][self.tickers[index][x]].loc[str(self.dates[self.current_iteration+600])]
            except Exception:
                two_year_value = 0

            return [self.date, curr_value, day_value, week_value, month_value,three_month_value ,six_month_value, year_value, two_year_value]

        else:
            curr_value =  self.open[self.tickers[x]].loc[str(self.dates[self.current_iteration])]
            day_value =  self.open[self.tickers[x]].loc[str(self.dates[self.current_iteration+1])]

            try:
                week_value =  self.open[self.tickers[x]].loc[str(self.dates[self.current_iteration+5])]
            except Exception:
                week_value = 0

            try:
                month_value =  self.open[self.tickers[x]].loc[str(self.dates[self.current_iteration+25])]
            except Exception:
                month_value = 0

            try:
                three_month_value =  self.open[self.tickers[x]].loc[str(self.dates[self.current_iteration+75])]
            except Exception:
                three_month_value = 0

            try:
                six_month_value =  self.open[self.tickers[x]].loc[str(self.dates[self.current_iteration+150])]
            except Exception:
                six_month_value = 0

            try:
                year_value =  self.open[self.tickers[x]].loc[str(self.dates[self.current_iteration+300])]
            except Exception:
                year_value = 0

            try:
                two_year_value =  self.open[self.tickers[x]].loc[str(self.dates[self.current_iteration+600])]
            except Exception:
                two_year_value = 0

            return [self.date, curr_value, day_value, week_value, month_value,three_month_value,six_month_value, year_value, two_year_value]
             
    def add_trades(self, buy, sell, s=0):
        """
        Add trade signal
        :param buy: list of stock tickers to buy
        :param long: list of stock tickers to sell
        """
        if self.multi_algs:
            if buy != "NA":
                buy = buy[0][0]
                new_position = self.add_trade(buy,s)
                if buy in self.buys.keys():
                    self.buys[buy].append(new_position)
                else: self.buys[buy] = [new_position]

            if sell != "NA":
                sell = sell[0][0]
                new_position = self.add_trade(sell,s)
                if sell in self.sells.keys():
                    self.sells[sell].append(new_position)
                else: self.sells[sell] = [new_position]

            if buy != "NA" and sell != "NA":
                new_position = [1+self.add_trade(buy,s)[i]/self.add_trade(buy,s)[1] - self.add_trade(sell,s)[i]/self.add_trade(sell,s)[1] for i in range(1,9)]

            elif buy != "NA":
                new_position = [self.add_trade(buy,s)[i]/self.add_trade(buy,s)[1] for i in range(1,9)]

            elif  sell != "NA":
                new_position = [self.add_trade(sell,s)[i]/self.add_trade(sell,s)[1] for i in range(1,9)]
            else:
                new_position = [0 for i in range(1,9)]

            self.comb.append(new_position)

        else:
            ''' Add to Buy data '''
            if buy != "NA":
                buy = buy[0][0]
                new_buy = self.add_trade(buy)

                if self.graph_signal:
                    if new_buy[3] != 0:
                        self.week_values.append((new_buy[3]/new_buy[1]-1))
                    else:
                        self.week_values.append(0)

                    if new_buy[4] != 0:
                        self.month_values.append((new_buy[4]/new_buy[1]-1))
                    else:
                        self.month_values.append(0)

                    if new_buy[5] != 0:
                        self.three_month_values.append((new_buy[5]/new_buy[1]-1))
                    else:
                        self.three_month_values.append(0)

                if buy in self.buys.keys():
                    self.buys[buy].append(new_buy)
                else: self.buys[buy] = [new_buy]

            ''' Add to Sell data '''
            if sell != "NA":
                sell = sell[0][0]
                new_sell = self.add_trade(sell)

                # if self.graph_signal:
                #     if new_sell[3] != 0:
                #         self.week_values.append((new_sell[3]/new_sell[1]-1))
                #     else:
                #         self.week_values.append(0)

                #     if new_sell[4] != 0:
                #         self.month_values.append((new_sell[4]/new_sell[1]-1))
                #     else:
                #         self.month_values.append(0)

                #     if new_sell[5] != 0:
                #         self.three_month_values.append((new_sell[5]/new_sell[1]-1))
                #     else:
                #         self.three_month_values.append(0)

                if sell in self.sells.keys():
                    self.sells[sell].append(new_sell)
                else: self.sells[sell] = [new_sell]


            ''' Add to Combined data '''
            if buy != "NA" and sell != "NA":
                new_position = [1+new_buy[i]/new_buy[1] - new_sell[i]/new_sell[1] for i in range(1,9)]

            elif buy != "NA":
                new_position = [new_buy[i]/new_buy[1] for i in range(1,9)]

            elif  sell != "NA":
                new_position = [new_sell[i]/new_sell[1] for i in range(1,9)]
            else:
                new_position = [0 for i in range(1,9)]

            self.comb.append(new_position)

    def buy_info(self):
        """
        Formats the buy information
        """

        performance = [0,0,0,0,0,0,0]
        prob_of_profit = [0,0,0,0,0,0,0]

        for i in range(2,9):
            num = 0
            p = 0
            prob = 0

            for key in self.buys:
                for signal in self.buys[key]:
                    if signal[i] == 0: continue
                    p += signal[i]/signal[1]
                    if signal[i]/signal[1] > 1:
                        prob += 1
                    num += 1

            try:
                performance[i-2] = p/num
                prob_of_profit[i-2] = prob/num
            except Exception:
                performance[i-2] = 0
                prob_of_profit[i-2] = 0

        return performance, prob_of_profit

    def sell_info(self):
        """
        Formats the sell information
        """

        performance = [0,0,0,0,0,0,0]
        prob_of_profit = [0,0,0,0,0,0,0]
        for i in range(2,9):
            num = 0
            p = 0
            prob = 0
            for key in self.sells:
                for signal in self.sells[key]:
                    if signal[i] == 0: continue
                    p += signal[i]/signal[1]
                    if signal[i]/signal[1] > 1:
                        prob += 1
                    num += 1

            try:
                performance[i-2] = p/num
                prob_of_profit[i-2] = prob/num
            except Exception:
                performance[i-2] = 0
                prob_of_profit[i-2] = 0

        return performance, prob_of_profit
    
    def joint_info(self):
        """
        Formats the both the buy and sell information
        """

        performance = [0,0,0,0,0,0,0]
        prob = [0,0,0,0,0,0,0]
        for i in range(1,8):
            n = 0
            for pos in self.comb:
                if pos[i] == 0: continue
                performance[i-1] += pos[i]
                n += 1

                if pos[i] > 1:
                    prob[i-1] += 1

            try:
                performance[i-1] /= n
                prob[i-1] /= n
            except Exception:
                performance[i-1] = 0
                prob[i-1] = 0

        return performance, prob

    def group_performace(self):
        """
        Overall performance for the index over time period.
        """

        tot = 0
        if self.multi_algs:
            l = sum([len(self.tickers[s]) for s in range(self.num_algs)])
            for s in range(self.num_algs):
                sect_tot = 0
                for tick in self.tickers[s]:
                    if self.open[s][tick].loc[str(self.dates[0])] != 0:
                        sect_tot += self.open[s][tick].loc[str(self.dates[-1])]/self.open[s][tick].loc[str(self.dates[0])]
                self.sector_performance[s] = (sect_tot-1)*100/len(self.tickers[s])
                tot += sect_tot

            return tot / l

        for tick in self.tickers:
            if self.open[tick].loc[str(self.dates[0])] != 0:
                tot += self.open[tick].loc[str(self.dates[-1])]/self.open[tick].loc[str(self.dates[0])]
        return tot / len(self.tickers)

    def plot(self, method, method_name = "Average Return"):
        """
        Plots overall algorithm performance over trading window.
        :return: plt plot of performance
        """ 
        
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(10)              

        mpl.rcParams['font.size'] = 15
        
        d, prob = method()

        for i in range(len(d)):
            d[i] = (d[i]-1)*100

        plt.plot(["Day", "Week", "Month", "3 Month", "6 Month", "Year", "2 Year"], d, label="Profit")
        plt.xlabel("Period After Signal")
        plt.ylabel("Percentage Increase")
        
        plt.legend()
        plt.savefig(method_name, format='svg')

    def plot_all(self, plot_pre=""):
        """
            Plots overall  algorithm performance over trading window for the buy, sell, and joint info.
            :return: plt plot of performance
            """ 

        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(10)              

        mpl.rcParams['font.size'] = 15
        plt.rcParams["figure.autolayout"] = True
        
        d1 = self.buy_info()[0]
        d2 = self.sell_info()[0]
        d3 = self.joint_info()[0]
        g = (self.group_performace()-1)*100

        for i in range(len(d1)):
            d1[i] = (d1[i]-1)*100
            d2[i] = (d2[i]-1)*100
            d3[i] = (d3[i]-1)*100

        plt.plot(["Day", "Week", "Month", "3 Month", "6 Month", "Year", "2 Year"], d1, label="BUY", linewidth=2.5,autoscale_on=False)
        plt.plot(["Day", "Week", "Month", "3 Month", "6 Month", "Year", "2 Year"], d2, label="SELL", linewidth=2.5,autoscale_on=False)
        plt.plot(["Day", "Week", "Month", "3 Month", "6 Month", "Year", "2 Year"], d3, label="JOINT", linewidth=2.5,autoscale_on=False)
        plt.plot(["Day", "Week", "Month", "3 Month", "6 Month", "Year", "2 Year"], [g for i in range(len(d1))], label="Sector Net Performance")
        plt.xlabel("Period After Signal")
        plt.ylabel("Percentage Increase")
        
        plt.legend()
        plt.savefig(plot_pre+"Comp")

    def nplot_all(self, plot_pre=""):
        """
            Plots overall algorithm performance normalized over trading window for buy, sell, and joint information.
            :return: plt plot of performance
            """ 

        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(10)              

        mpl.rcParams['font.size'] = 15
        
        d1 = self.buy_info()[0]
        d2 = self.sell_info()[0]

        g = (self.group_performace()**0.2-1)*100

        d1 = [(d1[0]**(250)-1)*100, (d1[1]**(250/5)-1)*100, (d1[2]**(250/25)-1)*100, (d1[3]**(250/75)-1)*100, (d1[4]**(2)-1)*100, (d1[5]**(1)-1)*100, (d1[6]**(1/2)-1)*100]
        d2 = [(d2[0]**(250)-1)*100, (d2[1]**(250/5)-1)*100, (d2[2]**(250/25)-1)*100, (d2[3]**(250/75)-1)*100, (d2[4]**(2)-1)*100, (d2[5]**(1)-1)*100, (d2[6]**(1/2)-1)*100]
        d3 = [d1[i] - d2[i] for i in range(7)]

        plt.plot(["Day", "Week", "Month", "3 Month", "6 Month", "Year", "2 Year"], d1, label="BUY", linewidth=2.5)
        plt.plot(["Day", "Week", "Month", "3 Month", "6 Month", "Year", "2 Year"], d2, label="SELL", linewidth=2.5)
        plt.plot(["Day", "Week", "Month", "3 Month", "6 Month", "Year", "2 Year"], d3, label="JOINT", linewidth=2.5)
        plt.plot(["Day", "Week", "Month", "3 Month", "6 Month", "Year", "2 Year"], [g for i in range(len(d1))], label="Sector CAG")
        plt.xlabel("Period After Signal")
        plt.ylabel("Percentage Increase")
        
        plt.legend()
        plt.savefig(plot_pre+"Norm")

    def plot_prob(self, plot_pre=""):

        """
            Plots the win probability for the buy, sell, and joint algorithms.
            :return: plt plot of performance
            """ 

        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(10)              

        mpl.rcParams['font.size'] = 15
        plt.rcParams["figure.autolayout"] = True
        
        d1 = self.buy_info()[1]
        d2 = self.sell_info()[1]
        d3 = self.joint_info()[1]

        d1 = [i*100 for i in d1]
        d2 = [i*100 for i in d2]
        d3 = [i*100 for i in d3]

        plt.bar(["Day", "Week", "Month", "3 Month", "6 Month", "Year", "2 Year"], d1, label="BUY")
        plt.bar(["Day", "Week", "Month", "3 Month", "6 Month", "Year", "2 Year"], d2, label="SELL")
        plt.bar(["Day", "Week", "Month", "3 Month", "6 Month", "Year", "2 Year"], d3, label="JOINT")

        plt.xlabel("Period After Signal")
        plt.ylabel("Win Probability (%")
        
        plt.legend()
        plt.savefig(plot_pre+"WP")

    def plot_vol(self, plot_pre=""):
        """
            Plots overall algorithm performance normalized over trading window for buy, sell, and joint information.
            :return: plt plot of performance
            """ 

        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(10)              

        mpl.rcParams['font.size'] = 15

        plt.plot(["Week", "Month", "3 Month"], [np.std(self.week_values), np.std(self.month_values), np.std(self.three_month_values)], label="Buy", linewidth=2.5)
        plt.plot(["Week", "Month", "3 Month"], [np.std([self.comb[i][2] for i in range(len(self.comb))]), np.std([self.comb[i][3] for i in range(len(self.comb))]),
         np.std([self.comb[i][4] for i in range(len(self.comb))])], label="Joint", linewidth=2.5)
        
        plt.legend()
        plt.savefig(plot_pre)
    
    def plot_signal(self, plot_pre=""):
        """
            Plots overall algorithm performance normalized over trading window for buy, sell, and joint information.
            :return: plt plot of performance
            """ 

        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(10)              

        mpl.rcParams['font.size'] = 15

        plt.scatter(self.buy_signals, self.week_values, label="Week", linewidth=2.5)
        plt.plot(sorted(self.buy_signals), self.best_fit(self.buy_signals, self.week_values), label="Week Fit")
        # plt.plot(sorted(self.buy_signals), self.best_fit(self.buy_signals, self.month_values), label="Month Fit")
        # plt.plot(sorted(self.buy_signals), self.best_fit(self.buy_signals, self.three_month_values), label="3 Month Fit")
        
        plt.legend()
        plt.savefig(plot_pre)

    def best_fit(self ,t_vals, v):
        A = [[t_vals[i], v[i]] for i in range(len(t_vals))]
        A = sorted(A,key=lambda x: x[0])

        t_vals = [A[i][0] for i in range(len(t_vals))]
        v = [A[i][1] for i in range(len(t_vals))]

        B =  [[t**i for i in range(6)] for t in t_vals]
        B = np.array(B)

        v = np.array(v)
        y =  lstsq(B,v)[0]

        return B@y*10