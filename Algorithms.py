import numpy as np

import tkinter as tk
import tkinter.scrolledtext as scrolledtext

from tkinter import ttk
from PIL import ImageTk, Image

class TradingAlgorithm():
    '''
    Parent Trading Algorithm. All algorithms are inherited from this class.
    '''

    def __init__(self,arr,ticks,mo="MACD",k1=2,k2=1,index="", sector_momentum = [], alpha_val=False, num_terms=10):
        """
        Trading Algorithm parent class. Contains most methods to be use in refined children algorithms.
        :param arr: Correlation Matrix.
        :param ticks: List of stock tickers.
        :param mo: Variable to use ("MACD", "Vol",...).
        :param k1: Number of most central stocks.
        :param k2: Number of stocks to buy and sell at a time.
        :param index: Index containing stocks to trade. Relevant for many algorithms.
        :param indices_momentum: Momentum of indices.
        :thresh: Threshold that must be surpassed in order to make trade.
        """
        self.k1 = k1
        self.k2 = k2
        self.momentum = []
        self.alg_name="Sector Switch"
        self.m=mo
        self.arr=arr
        self.tickers = ticks
        self.num_stocks = len(ticks)
        self.index = index
        self.sector_momentum = sector_momentum
        self.num_terms = num_terms

        if len(sector_momentum) > 0:
            self.num_stocks -= 1

        self.alpha_val = alpha_val

        self.buy_signal = 0
        self.sell_signal = 0

        self.leig = 0
        
    def alpha(self,date,dataframe,iteration):
        """
        Computes buy, sell signals for backtest using algorithm strategy.
        :param date: Current date to trade on.
        :param dataframe: Dataframe containing open price data.
        :param iteration: Current iteration in backtest.
        :return: Buy and Sell signals (each a lists of stock tickers)
        """
        self.date=date
        self.df=dataframe # construct graph
        self.adj=self.arr[str(date)]
        self.calc_momentum(dataframe) # calculate momentum
        b1,s1=self.strategy()
        
        return self.buy_format(b1), self.buy_format(s1), [self.buy_signal, self.sell_signal]
    
    def index_momentum(self):
        """
        Locates index momentum from given dataset.
        :return: Index momentum at current date.
        """
        return self.sector_momentum[self.date]

    def calc_momentum(self,dataframe):
        """
        Computes momentum using open prices over time period
        :param dataframe: dataframe containing momentum values
        """

        self.momentum = []
        for t in self.tickers:
            if t != "Index":
                self.momentum.append(dataframe[t+self.m].iloc[-1])
        
    def buy_format(self,l,weights=[]):
        """
        Formats buy/sell signals for desired format in backtest.
        :param l: list of stocks to buy/sell
        :param weights: list of weights for each stock (default is equal allocation)
        :return: Formatted buy and sell signals (list of tuples (stock, percentage to allocate))
        """
        if weights: return [(l[i], l.count(l[i])*weights[i]) for i in range(len(l))]
        else: return [(l[i], l.count(l[i])/len(l)/2) for i in range(len(l))]
    
    def largest_elements(self,lst,v=0):
        """
        Computes buy, sell signals for backtest using algorithm strategy.
        :param lst: List to find largest values from.
        :param v: *Fill in*
        :return: k2 largest elements of list (default 1)
        """
        if v == 0: return np.flip(np.argsort(lst)[-1*self.k:])
        else: return np.flip(np.argsort(lst)[-1*v:])
    
    def smallest_elements(self,lst,v=0):
        """
        Computes buy, sell signals for backtest using algorithm strategy.
        :param lst: List to find smallest values from.
        :param v: *Fill in*
        :return: k2 smallest elements of list (default 1)
        """
        if v == 0: return np.argsort(lst)[:self.k]
        else: return np.argsort(lst)[:v]

    def index_outlook(self):
        """
        Computes Outlook of index using centrality (measure of importance) * momentum for each stock
        :return: index outlook
        """
        return [self.lev[i]*self.momentum[i] for i in range(self.num_stocks)]
    
    def deg_centrality(self,A,absolute=False):
        """
        Computes basic degree centrality
        :return: list degree centrality
        """
        if absolute: 
            return [sum([abs(A[i][j]) for j in range(self.num_stocks)]) for i in range(self.num_stocks)]
        else: return [sum(A[i]) for i in range(self.num_stocks)]

    def shortest_path(self, A, i, j):
        positions = np.where(A == 3)
        x, y = positions

        dx = abs(i - x)  # Horizontal distance
        dy = abs(j - y)  # Vertical distance

        # There are different criteria to compute distances
        euclidean_distance = np.sqrt(dx ** 2 + dy ** 2)
        my_distance = euclidean_distance  # Criterion choice
        min_dist = min(my_distance)

        return min_dist

    def closeness_centrality(self,A):
        x = np.zeros(self.num_stocks)
        for i in range(self.num_stocks):
            dist = sum(self.shortest_path(A,i,j) for j in range(self.num_stocks))
            print(dist)
            x[i] = (self.num_stocks-1) / dist if dist != 0 else 0

        return x

    def calc_pr(self, A, alpha=0.35, pos=3):
        """
        Calculates Page Rank Centrality
        :param A: adjacency matrix
        :param alpha: alpha used in page rank formula
        :return: list of page rank centrality values for each stock
        """

       # Default - Used in testing
        if pos==1:

            if self.alpha_val != False:
                alpha = self.alpha_val

            sum = 0
            D = np.identity(self.num_stocks) # Diagonal Matrix with entries 1/deg_i
            e = np.ones(self.num_stocks) # 1 vector
            adj = np.identity(self.num_stocks)
            for k in range(self.num_terms):
                adj = adj@A
                d = self.deg_centrality(adj, absolute=True) # update the degree centrality
                for i in range(self.num_stocks):
                    if d[i] == 0: D[i][i] = 0
                    else: D[i][i] = 1/abs(d[i])
                sum += alpha**(k+1)*adj@D

            return sum @ e

        # Same as 1 however multiplies by (1-alpha)v
        elif pos==2:

            if self.alpha_val != False:
                alpha = self.alpha_val

            sum = 0
            D = np.identity(self.num_stocks) # Diagonal Matrix with entries 1/deg_i
            e = np.ones(self.num_stocks) # 1 vector
            v = np.ones(self.num_stocks) / self.num_stocks # 1 vector

            d = self.deg_centrality(A, absolute=True) # update the degree centrality
            for i in range(self.num_stocks):
                    if d[i] == 0: D[i][i] = 0
                    else: D[i][i] = 1/abs(d[i])

            P = A@D

            adj = alpha*P + (1-alpha)*np.outer(v,e)
            eig = np.linalg.eig(adj)[1][0]

            return  eig

        # PageRank Iteration X_k+1 - alpha P x_k + (1- alpha)v
        elif pos==3:
            if self.alpha_val != False:
                alpha = self.alpha_val
                
            D = np.identity(self.num_stocks) # Diagonal Matrix with entries 1/deg_i
            v = np.ones(self.num_stocks)/self.num_stocks
            x = np.ones(self.num_stocks)/self.num_stocks

            d = self.deg_centrality(A, absolute=True) # update the degree centrality
            for i in range(self.num_stocks):
                    if d[i] == 0: D[i][i] = 0
                    else: D[i][i] = 1/abs(d[i])

            P = A@D

            for k in range(self.num_terms):
                x = alpha * P@x + v #(1-alpha)*v

            # if np.min(x) <= 0:
            #     print(np.min(x))

            # print(np.sum(np.abs(x)))

            return x

    def calc_katz(self, A, alpha=0.45):
        e = np.ones(self.num_stocks)
        I = np.identity(self.num_stocks)
        return (np.linalg.inv(I - alpha*A.T)-I)@e

#  TO DO : Add in sector info. When certain sector indicator is met, buy/sell sector and sell/buy stocks
class LinkMomentum(TradingAlgorithm):
    '''
    Link Momentum. Computes stocks to buy and sell based on momentum relative to high correlation stocks.
    '''

    def __init__(self,arr,ticks,k2=1,method="dif",alpha_val=False,num_terms=15):
        """
        Link Momentum. Computes stocks to buy and sell based on momentum relative to high correlation stocks.
        :param arr: Correlation Matrix
        :param ticks: List of stock tickers
        :param k2: Number of stocks to buy at a time.
        :param method: Method to use in calculating rankings to buy/sell. "dif" : momentum difference, "cor" : pure momentum, "absdif" : absolute difference
        """

        super().__init__(arr,ticks,k2=k2,alpha_val=alpha_val, num_terms=num_terms)
        self.method = method
        
    def strategy(self):
        """
        Implementation of Strategy
        """

        m =  self.method_function()
        a = self.calc_pr(m)

        a_l = [a[j] if int(self.df[self.tickers[j]].iloc[-1])!=0 else -1000 for j in range(len(a))]
        a_s = [a[j] if int(self.df[self.tickers[j]].iloc[-1])!=0 else 1000 for j in range(len(a))]

        self.buy_signal = max(a_l)
        self.sell_signal = min(a_s)

        return [np.argmax(a_l)], [np.argmin(a_s)]

    def method_function(self):
        """
        Method Function dictated by initialization variable "method".
        """
        if self.method == "dif":  # momentum difference
            return np.array([[self.adj[i][j]*(self.momentum[j]-self.momentum[i]) if j!=i else self.momentum[i] for j in range(self.num_stocks)] for i in range(self.num_stocks)])
        elif self.method == "imdif":  # improved momentum difference (multiplies by the momentum magnitude)
            return np.array([[self.adj[i][j]*(self.momentum[j]-self.momentum[i])*(-self.momentum[j]) for j in range(self.num_stocks)] for i in range(self.num_stocks)])
        elif self.method == "cor": # pure momentum
            return np.array([[self.adj[i][j]*(self.momentum[j]) for j in range(self.num_stocks)] for i in range(self.num_stocks)])
        elif self.method =='absdif':
            return np.array([[abs(self.adj[i][j])*abs(self.momentum[j]-self.momentum[i]) for j in range(self.num_stocks)] for i in range(self.num_stocks)])
        elif self.method == 'vol':
            return np.array([[abs(self.adj[i][j])*abs(self.momentum[j]-self.momentum[i]) for j in range(self.num_stocks)] for i in range(self.num_stocks)])

    def most_central(self):
        """
        Compute most central stock
        """
        most_central = self.largest_elements(self.calc_pr(self.adj), 5) # indexes of most central stocks
        pass
   
# TO DO : Need to get Sentiment/Analyst Data
class LinkSentiment(TradingAlgorithm):
    '''
    Link Sentiment. Computes stocks to buy and sell based on sentiment relative to high correlation stocks.
    '''

    def __init__(self,sentiment,arr,ticks,k2=1, method="dif"):
        """
        Link Sentiment. Computes stocks to buy and sell based on sentiment relative to high correlation stocks.
        :param sentiment: Sentiment Dataset
        :param arr: Correlation Matrix
        :param ticks: List of stock tickers
        :param k2: Number of stocks to buy at a time.
        :param method: Method to use in calculating rankings to buy/sell. "dif" : sentiment difference, "cor" : pure sentiment
        """
        super().__init__(arr,ticks,mo="MACD",k1=2,k2=1)
        self.sentiment = sentiment
        self.method = method

    def get_sentiment(self):
        """
        Gets sentiment data from dataset
        :return: Sentiment data for current date.
        """
        return self.sentiment[self.date]

    def strategy(self):
        """
        Implementation of strategy.
        """
        m = self.method_function()
        a = self.calc_pr(m)
        a_l = [a[j] if self.df[self.tickers[j]].iloc[-1]!=0 else -1000 for j in range(len(a))]
        a_s = [a[j] if self.df[self.tickers[j]].iloc[-1]!=0 else 1000 for j in range(len(a))]
        return [i for i in self.largest_elements(a_l, self.k2)], []

    def method_function(self):
        """
        Method Function dictated by initialization variable "method".
        """
        if self.method == "dif":  # sentiment difference
            return np.array([[self.adj[i][j]*(self.get_sentiment()[j]-self.get_sentiment()[i]) for j in range(self.num_stocks)] for i in range(self.num_stocks)])
        elif self.method == "cor": # pure sentiment
            return np.array([[self.adj[i][j]*(self.get_sentiment()[j]) for j in range(self.num_stocks)] for i in range(self.num_stocks)])

# TO DO : Need to implement Index Data for trade (add to dataframe op/df for backtest method)
class SectorMomentum(TradingAlgorithm):
    '''
    Link Momentum. Computes stocks to buy and sell based on momentum relative to high correlation stocks.
    '''

    def __init__(self,arr,ticks,k2=1,method="dif",alpha_val=False, sector_momentum=[]):
        """
        Link Momentum. Computes stocks to buy and sell based on momentum relative to high correlation stocks.
        :param arr: Correlation Matrix
        :param ticks: List of stock tickers
        :param k2: Number of stocks to buy at a time.
        :param method: Method to use in calculating rankings to buy/sell. "dif" : momentum difference, "cor" : pure momentum, "absdif" : absolute difference
        """
        super().__init__(arr,ticks,k2=k2,alpha_val=alpha_val,sector_momentum=sector_momentum)
        self.method = method
        
    def strategy(self):
        """
        Implementation of Strategy
        """

        a = self.calc_pr(self.method_function())

        a_l = [a[j] if int(self.df[self.tickers[j]].iloc[-1])!=0 else -1000 for j in range(len(a))]
        a_s = [a[j] if int(self.df[self.tickers[j]].iloc[-1])!=0 else 1000 for j in range(len(a))]

        most_central = np.argmax(a_l)
        least_central = np.argmin(a_s)

        self.buy_signal = max(a_l)
        self.sell_signal = min(a_s)


        if int(self.df[self.tickers[-1]].iloc[-1])!=0:

            if self.index_momentum() <= 0: s = [least_central]
            else: s = [self.num_stocks]

            return [most_central], s

        else:
            return [most_central], [least_central]


    def method_function(self):
        """
        Method Function dictated by initialization variable "method".
        """
        if self.method == "dif":  # momentum difference
            return np.array([[self.adj[i][j]*(self.momentum[j]-self.momentum[i]) for j in range(self.num_stocks)] for i in range(self.num_stocks)])
        elif self.method == "imdif":  # improved momentum difference (multiplies by the momentum magnitude)
            return np.array([[self.adj[i][j]*(self.momentum[j]-self.momentum[i])*(abs(self.momentum[j])+abs(self.momentum[i])) for j in range(self.num_stocks)] for i in range(self.num_stocks)])
        elif self.method == "cor": # pure momentum
            return np.array([[self.adj[i][j]*(self.momentum[j]) for j in range(self.num_stocks)] for i in range(self.num_stocks)])
        elif self.method =='absdif':
            return np.array([[abs(self.adj[i][j])*abs(self.momentum[j]-self.momentum[i]) for j in range(self.num_stocks)] for i in range(self.num_stocks)])
        elif self.method == 'vol':
            return np.array([[abs(self.adj[i][j])*abs(self.momentum[j]-self.momentum[i]) for j in range(self.num_stocks)] for i in range(self.num_stocks)])

# TO DO : Everything
class SectorLinkMomentum(TradingAlgorithm):
    '''
    Combination of Sector and Link Momentum. Computes stocks to buy and sell based on momentum relative to high correlation stocks. Computes over given sectors and 
    weighs high/low values stocks within sectors according to sector momentum.
    '''

    def __init__(self,sectors,indices_momentum,arr,ticks,k1=4,k2=1,thresh=0):
        """
        Combination of Sector and Link Momentum. Computes stocks to buy and sell based on momentum relative to high correlation stocks. Computes over given sectors and 
        weighs high/low values stocks within sectors according to sector momentum.
        :param sectors: List of sector indices to consider stocks within.
        :param indices_momentum: Index momentum dataset.
        :param arr: Correlation Matrix.
        :param ticks: List of stock tickers.
        :param k1: Number of most central stocks.
        :param k2: Number of stocks to buy and sell at a time.
        :thresh: Threshold that must be surpassed in order to make trade.
        """
        super().__init__(arr,ticks,mo="MACD",index=sectors,indices_momentum=indices_momentum,k1=k1, k2=k2)

    def strategy(self):
        """
        Implementation of strategy.
        """
        pass

# Main Method
class MultipleLinkMomentum(TradingAlgorithm):

    def __init__(self,arr,ticks,k2=1,method="dif",num_sectors = 1):
        """
        Multiple Link Momentum. Computes stocks to buy and sell based on momentum relative to high correlation stocks. Does so for a set of
        Sectors/Groups and then returns the highest valued buy/sell signals amongst sector signals.
        :param arr: List of Correlation Matrix (for each sector s)
        :param ticks: Set of Lists of stock tickers (for each sector s)
        :param k2: Number of stocks to buy at a time.
        :param method: Method to use in calculating rankings to buy/sell. "dif" : momentum difference, "cor" : pure momentum, "absdif" : absolute difference
        """

        self.method = method
        self.num_stocks = num_sectors

        for s in range(self.num_sectors):
            self.algs.append(LinkMomentum(arr[s],ticks[s],k2=1,method="dif"))

    def alpha(self,date,dataframe,iteration):
        buy_signals = []
        sell_signals = []

        for s in range(self.num_sectors):
            b,s, = self.algs[s].alpha(date,dataframe,iteration) # get list of buy and sell signal
            buy_signals += b
            sell_signals += s

        return buy_signals, sell_signals, []

# Momentum Portfolio
class MomentumPortfolio(TradingAlgorithm):
    def __init__(self,arr,ticks):
            """
            Link Momentum. Computes stocks to buy and sell based on momentum relative to high correlation stocks.
            :param arr: Correlation Matrix
            :param ticks: List of stock tickers
            :param k2: Number of stocks to buy at a time.
            :param method: Method to use in calculating rankings to buy/sell. "dif" : momentum difference, "cor" : pure momentum, "absdif" : absolute difference
            """

            super().__init__(arr,ticks,k2=0,alpha_val=0, num_terms=0)
            
    def strategy(self):
        """
        Implementation of Strategy
        """
        quintile_len =int(self.num_stocks/10)
        a_l = self.largest_elements([self.momentum[j] if int(self.df[self.tickers[j]].iloc[-1])!=0 else -100 for j in range(self.num_stocks)], quintile_len)
        a_s = self.smallest_elements([self.momentum[j] if int(self.df[self.tickers[j]].iloc[-1])!=0 else 100 for j in range(self.num_stocks)], quintile_len)



        self.buy_signal = 1
        self.sell_signal = -1

        return [el for el in a_l], [el for el in a_s]


class LinkPortfolio(TradingAlgorithm):
    '''
    Link Momentum. Computes stocks to buy and sell based on momentum relative to high correlation stocks.
    '''

    def __init__(self,arr,ticks,k2=1,method="dif",alpha_val=False,num_terms=15, r=1):
        """
        Link Momentum. Computes stocks to buy and sell based on momentum relative to high correlation stocks.
        :param arr: Correlation Matrix
        :param ticks: List of stock tickers
        :param k2: Number of stocks to buy at a time.
        :param method: Method to use in calculating rankings to buy/sell. "dif" : momentum difference, "cor" : pure momentum, "absdif" : absolute difference
        """

        super().__init__(arr,ticks,k2=k2,alpha_val=alpha_val, num_terms=num_terms)
        self.method = method
        self.rank=r
        
    def strategy(self):
        """
        Implementation of Strategy
        """

        m =  self.method_function()
        a = self.calc_pr(m)

        a_l = [a[j] if int(self.df[self.tickers[j]].iloc[-1])!=0 else -1000 for j in range(len(a))]
        a_s = [a[j] if int(self.df[self.tickers[j]].iloc[-1])!=0 else 1000 for j in range(len(a))]

        self.buy_signal = max(a_l)
        self.sell_signal = min(a_s)
        quintile_len =int(self.num_stocks/10)

        self.buy_signal = 1
        self.sell_signal = -1

        if self.rank > 1:
            r1 = set([el for el in self.largest_elements(a_l,(self.rank-1)*quintile_len)])
            r2 = set([el for el in self.largest_elements(a_l,self.rank*quintile_len)])
            x = list(r2-r1)
            r=[]

            for i in x:
                if int(self.df[self.tickers[i]].iloc[-1])!=0:
                    r.append(i)


            # l1 = set([el for el in self.smallest_elements(a_l,(self.rank-1)*quintile_len)])
            # l2 = set([el for el in self.smallest_elements(a_s,self.rank*quintile_len)])
            # x = list(l2-l1)
            # l=[]
            
            # for i in x:
            #     if int(self.df[self.tickers[i]].iloc[-1])!=0:
            #         l.append(i)


        else:
            r = [el for el in self.largest_elements(a_l,quintile_len)]
            # l = [el for el in self.smallest_elements(a_s,quintile_len)]

        l = []
        return list(r), list(l)

    def method_function(self):
        """
        Method Function dictated by initialization variable "method".
        """
        if self.method == "dif":  # momentum difference
            return np.array([[self.adj[i][j]*(self.momentum[j]-self.momentum[i]) if j!=i else self.momentum[i] for j in range(self.num_stocks)] for i in range(self.num_stocks)])
        elif self.method == "imdif":  # improved momentum difference (multiplies by the momentum magnitude)
            return np.array([[self.adj[i][j]*(self.momentum[j]-self.momentum[i])*(-self.momentum[j]) for j in range(self.num_stocks)] for i in range(self.num_stocks)])
        elif self.method == "cor": # pure momentum
            return np.array([[self.adj[i][j]*(self.momentum[j]) for j in range(self.num_stocks)] for i in range(self.num_stocks)])
        elif self.method =='absdif':
            return np.array([[abs(self.adj[i][j])*abs(self.momentum[j]-self.momentum[i]) for j in range(self.num_stocks)] for i in range(self.num_stocks)])
        elif self.method == 'vol':
            return np.array([[abs(self.adj[i][j])*abs(self.momentum[j]-self.momentum[i]) for j in range(self.num_stocks)] for i in range(self.num_stocks)])

    def most_central(self):
        """
        Compute most central stock
        """
        most_central = self.largest_elements(self.calc_pr(self.adj), 5) # indexes of most central stocks
        pass