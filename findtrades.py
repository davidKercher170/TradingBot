import pandas as pd
import yfinance as yf
import math
import numpy as np
import pickle

from scipy.stats import spearmanr

tech = ['GOOG', 'META', 'ACN', 'CSCO', 'NFLX', 'CRM', 'ORCL', 'PYPL', 'INTU', 'IBM', 'FISV', 'FIS', 'ADSK',
'SNPS', 'INFN', 'CDNS', 'MSCI', 'PAYX', 'GPN', 'MTCH', 'EPAM', 'VRSK', 'ANSS', 'TWTR', 'VRSN', 'CDW', 'CERN',
'IT', 'PAYC', 'TYL', 'BR', 'AKAM', 'FDS', 'CDAY', 'NLOK', 'PTC', 'JKHY', 'LDOS', 'CTXS', 'JNPR', 'DXC']
# Communication Services /Electric - XLC
elect = ['AAPL', 'NVDA', 'AVGO', 'INTC', 'QCOM', 'AMD', 'RTX', 'BA', 'MU', 'LRCX', 'LMT', 'ADI', 'KLAC',
'NXPI', 'NOC', 'GD', 'FTNT', 'TEL', 'APH', 'MCHP', 'MSI', 'ANET', 'LHX', 'HPQ', 'KEYS', 'TDG', 'GLW',
'ZBRA', 'TER', 'SWKS', 'FTV', 'GRMN', 'STX', 'ENPH', 'MPWR', 'TRMB', 'HPE', 'TDY', 'WDC', 'NTAP', 'QRVO',
'TXT', 'FFIV', 'HWM', 'IPGP', 'HII']
# Finance - XLF
fin = ['BRK-B', 'JPM', 'BAC', 'WFC', 'SPGI', 'MS', 'GS', 'SCHW', 'C', 'AXP', 'BLK', 'CB', 'MMC',
'PGR', 'CME', 'PNC', 'TFC', 'USB', 'AON', 'ICE', 'MCO', 'MET', 'AIG', 'COF', 'TRV', 'AJG', 'MSCI',
'AFL', 'PRU', 'ALL', 'MTF', 'BK', 'AMP', 'DFS', 'FRC', 'STT', 'TROW', 'FITB', 'WTW', 'SIVB', 'HIG', 
'RKF', 'RF', 'HBAN', 'NTRS', 'CFG', 'PFG', 'FDS', 'KEY', 'SYF', 'CINF', 'BRO', 'WRB', 'CBOE', 'SBNY',
'RE', 'L', 'CMA', 'MKTX', 'GL', 'ZION', 'AIZ', 'LNC', 'BEN', 'IVZ']
# Health Care - XLV
health = ["UNH", "JNJ", "PFE", "ABBV", "LLY", "MRK", "TMO", "ABT", "DHR", "BMY", "CVS", "AMGN", "MDT", "ELV", "CI",
 "GILD", "SYK","ISRG", "ZTS", "REGN", "BDX", "VRTX", "BSX", "EW", "HUM", "MCK", "CNC", "MRNA", "HCA", "IQV", "A",
 "DXCM", "RMD", "ILMN", "BIIB", "BAX"]
# Industrial - XLI
ind = ["UPS", "UNP", "RTZ", "HON", "DE", "LMT", "CAT", "BA", "GE", "NOC", "CSX", "WM", "MMM", "ETN", "NSC", "ITW", "GD",
 "FDX", "EMR", "LHX", "JCI", "TT", "CTAS","PH", "CARR", "TDG", "RSG", "CMI", "PCAR", "OTIS", "VRSK", "ROK", "AME", "FAST",
  "ODFL", "CPRT", "GWW", "EFX", "FTV", "LUV", "DAL", "URI", "PWR", "IR","DOV", "EXPD", "WAB", "IEX", "J"]
# Energy - XLE
energy = ["XOM", "CVX", "COP", "EOG", "OXY", "SLB", "PXD", "MPC", "VLO", "DVN", "PSX", "WMB", "KMI", "HES", "HAL", "CTRA",
 "FANG", "BKR", "MRO", "APA"]
# Real Estate - XLRE
re = ["AMT", "PLD", "CCI", "EQIX", "PSA", "O", "SBAC", "WELL", "DLR", "SPG", "VICI", "AVB", "EXR", "CBRE", "EQR", "WY", "ARE",
 "DRE", "VTR", "MAA", "ESS","IRM", "PEAK", "CPT", "UDR", "HST", "KIM", "BXP", "REG", "FRT"]

# get data for given index
def get_data(index):
    op = pd.DataFrame()
    df = pd.DataFrame()
    corr = pd.DataFrame()

    for i in index:
        d = yf.download(i, period="5y")
        op[i] = d["Open"]
        df[i] = d["Close"]
        corr[[i+"Open", i+"High", i+"Low", i+"Close"]] = d[["Open", "High", "Low", "Close"]]

    return op,df,corr

# normalize correlation values
def norm_corr(corr):
    num_cols = len(corr.columns)
    num_rows = len(corr.index)

    for col in corr.columns:
        x = [0]
        for i in range(1,num_rows):
            try:
                x.append(math.log(corr[col][i] / corr[col][i-1]))
            except Exception:
                x.append(0)

        corr[col] = x

    return corr

# open correlation matrix
def form_corr(corr, index,name,corr_length):
    open_prices = corr[[i+"Open" for i in index]]
    num_cols = len(open_prices.columns)

    a = np.zeros((num_cols,num_cols))
    r = open_prices.iloc[-1*corr_length:]

    for j in range(num_cols):
        x = r.iloc[:, j] # jth column
        for k in range(j+1, num_cols):
            y = r.iloc[:, k] # kth column
            rho, p = spearmanr(x, y)

            if str(rho) == 'nan':
                a[j][k], a[k][j] = 0, 0
            else:
                a[j][k], a[k][j] = rho, rho

    return a

# edit open entries
def edit_open(op):
    mom = []
    for i in op.columns:
        exp1 = op[i].ewm(span=12, adjust=False).mean()
        exp2 = op[i].ewm(span=26, adjust=False).mean()
        mom.append(exp1[-1]/exp2[-1] - 1)
    return mom

# complete data loop to organize data for a sector
def data_loop(indices,name=""):
    momentum = []
    arr = []
    opens = []

    for index in indices:
        op,df,corr = get_data(index)
        corr = norm_corr(corr)
        momentum.append(edit_open(op))
        arr.append(form_corr(corr,index,name,500))
        opens.append(op.fillna(0))

    with open("Open.pkl", 'wb') as file:
            pickle.dump(opens, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open("Momentum.pkl", 'wb') as file:
            pickle.dump(momentum, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open("Corr.pkl", 'wb') as file:
            pickle.dump(arr, file, protocol=pickle.HIGHEST_PROTOCOL)

def method(adj, momentum):
    num_stocks = len(adj)
    return np.array([[0 if j==i or adj[i][j]=='nan' or momentum[i]-momentum[j]==0 else adj[i][j]*(momentum[j]-momentum[i])/abs(momentum[i]+momentum[j]) for j in range(num_stocks)] for i in range(num_stocks)])

def lev(adj):
     num_stocks = len(adj)
     return [sum([adj[i][j] for j in range(num_stocks)]) for i in range(num_stocks)]

def pr(adj, alpha=0.45):
    for i in range(len(adj)):
        for j in range(len(adj)):
            if str(adj[i][j]) =='nan': 
                adj[i][j] = 0
    num_stocks = len(adj)


    d = [sum(adj[i]) for i in range(num_stocks)] # update the degree centrality
    D = np.identity(num_stocks) # Diagonal Matrix with entries 1/deg_i
    e = np.ones(num_stocks) # 1 vector

    for i in range(num_stocks): 
        if round(d[i],4) != 0: D[i][i] = 1/abs(d[i])
        else: D[i][i] = 1 # set Diagonal entries to 1/deg_i

    P = adj@D

    return np.linalg.inv(np.identity(num_stocks)-alpha*np.array(P))@e

def largest_elements(lst,k):
        """
        Computes buy, sell signals for backtest using algorithm strategy.
        :param lst: List to find largest values from.
        :param v: *Fill in*
        :return: k2 largest elements of list (default 1)
        """
        return np.flip(np.argsort(lst)[-1*k:])
    
def smallest_elements(lst,k):
    """
    Computes buy, sell signals for backtest using algorithm strategy.
    :param lst: List to find smallest values from.
    :param v: *Fill in*
    :return: k2 smallest elements of list (default 1)
    """
    return np.argsort(lst)[:k]

def trade(sectors = ["Tech", "Elec", "Fin", "Health", "Indust", "Energy", "REstate"]):
    num_sectors = len(sectors)
    b, s = [], []
    tickers = []

    with open("Corr.pkl", 'rb') as handle:
        A=pd.read_pickle(handle)

    with open("Momentum.pkl", 'rb') as handle:
        m=pd.read_pickle(handle)

    with open("Open.pkl", 'rb') as handle:
        open_data=pd.read_pickle(handle)

    for i in range(num_sectors):
        tickers.append(open_data[i].columns)

        mom = [m[i][j] if str(m[i][j]) != "nan" else 0 for j in range(len(m[i]))]
    
        B = method(A[i],mom)
        x = lev(B)
        b.append(np.argmax(x))
        s.append(np.argmin(x))

        p1 = str(round(open_data[i][tickers[i][b[i]]].iloc[-1],2))
        p2 = str(round(open_data[i][tickers[i][s[i]]].iloc[-1],2))
        print(str(sectors[i]) + " Buy : " + str(tickers[i][b[i]]) + "  Price: " + p1 + "  Signal: " + str(100*round(np.max(x)/np.linalg.norm(x,2),2))+"%")
        # print(str(sectors[i]) + " Sell : " + str(tickers[i][s[i]]) + "  Price: " + p2 + "  Signal: " + str(round(np.min(x),3)))

# data_loop([tech, elect, fin, health, ind, energy, re]) 
trade()