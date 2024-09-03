import pandas as pd
import yfinance as yf
import math
import numpy as np

from scipy.stats import spearmanr


tech, elect, fin, health, ind, energy, re = [], [], [], [], [], []
sectors = ["XLK", "XLC", "XLF", "XLV", "XLI", "XLE", "XLRE"]

# list of all indices
indices = [tech, elect, fin, health, ind, energy, re]
names = ["Tech", "Elect", "Fin", "Health", "Ind", "Energy", "REstate"]
sp = ["^GSPC"]

# get data for given index
def get_data(index, corr_len, date="2017-10-27"):
    op = pd.DataFrame()
    for i in index:
        d = yf.download(i, start=date-corr_len, end=date)
        op[i] = d["Open"]
    return op


# normalize correlation values
def norm_corr(op):
    num_rows = len(op)
    corr = pd.DataFrame()

    for col in op.columns:
        x = [0]
        for i in range(1,num_rows):
            try:
                x.append(math.log(op[col][i] / op[col][i-1]))
            except Exception:
                x.append(0)
        corr[col] = x

    return corr


# open correlation matrix
def form_corr(corr, index):
    open_prices = corr[[i+"Open" for i in index]]
    num_cols = len(open_prices.columns)

    a = np.zeros((num_cols,num_cols))

    for j in range(num_cols):
        x = open_prices.iloc[:, j] # jth column
        for k in range(j+1, num_cols):
            y = open_prices.iloc[:, k] # kth column
            rho, p = spearmanr(x, y)
            if str(rho) == 'nan':
                a[j][k], a[k][j] = 0, 0 
            else:
                a[j][k], a[k][j] = rho, rho

    return a


# edit open entries
def calc_macd(op):
    m = []
    for i in op.columns:
        exp1 = op[i].ewm(span=12, adjust=False).mean()[-1]
        exp2 = op[i].ewm(span=26, adjust=False).mean()[-1]
        m.append(exp1/exp2 - 1)
    return m


def deg_centrality(self,A):
        """
        Computes basic degree centrality
        :return: list degree centrality
        """
        return [sum([abs(A[i][j]) for j in range(self.num_stocks)]) for i in range(self.num_stocks)]


def pr_centrality(A, num_stocks, momentum, alpha=0.85, num_terms=15):
    sum = 0
    D = np.identity(num_stocks) # Diagonal Matrix with entries 1/deg_i
    e = np.ones(num_stocks) # 1 vector
    A = np.array([[A[i][j]*(momentum[j]-momentum[i]) for j in range(num_stocks)] for i in range(num_stocks)])
    adj = np.identity(num_stocks)

    for k in range(1,num_terms):
        adj = adj@A
        d = deg_centrality(adj) # update the degree centrality
        for i in range(num_stocks):
            if d[i] == 0: D[i][i] = 0
            else: D[i][i] = 1/abs(d[i])
        sum += alpha**k*adj@D

    return sum @ e

# complete data loop to organize data for a sector
def data_loop(index, corr_len, starting_date):
    d = starting_date
    for ind in indices:
        open_prices = get_data(index, corr_len, date=d)
        correlation_matrix = norm_corr(open_prices)
        correlation_matrix = form_corr(correlation_matrix)
        momentum = calc_macd(open_prices)
        x = pr_centrality(A = correlation_matrix, num_stocks=len(momentum))

        buy = np.argmax(x)
        sell = np.argmin(x)

