import tkinter
import tkinter as tk
import tkinter.scrolledtext as scrolledtext

from tkinter import ttk
from PIL import ImageTk, Image

class BacktestDisplay():

    def __init__(self,bt,bg='white'):
        """
        Displays information from backtest.
        :param bt: Backtest class
        :param bg: Background color
        """
        self.bg = bg
        self.bt = bt
        
        self.totret = str(round(bt.comp_ret(),2)) + "%"
        self.compret = str(round(bt.comp_annual(),2)) + "%"
        self.sharpe_val = str(round(bt.sharpe(),2))

        self.window = tk.Tk()
        self.window.title('Backtest')

        self.img = ImageTk.PhotoImage(Image.open(bt.plot()).resize((900,530)).crop([70,50,820,520]))
        
        if bt.adv_plots == True:
            self.indimg = ImageTk.PhotoImage(Image.open(bt.indplot()).resize((900,530)).crop([70,50,820,520]))
            self.bvolimg = ImageTk.PhotoImage(Image.open(bt.plot_bvol()).resize((900,530)).crop([70,50,820,540]))
            self.svolimg = ImageTk.PhotoImage(Image.open(bt.plot_svol()).resize((900,530)).crop([70,50,820,540]))
        
    def draw(self):
        """
        Draws display.
        """
        self.leftframe = tk.Frame(master=self.window, width=500,height=800,bg=self.bg,borderwidth=1)
        self.leftframe.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)
        
        self.rightframe = tk.Frame(master=self.window, width=500,height=800,bg=self.bg,borderwidth=1)
        self.rightframe.pack(side=tk.RIGHT,expand=True,fill=tk.BOTH)
        
        #  Graph Window with Tabs(TOP LEFT)
        self.notebook=ttk.Notebook(self.leftframe)
        self.notebook.pack(side=tk.TOP,expand=True,fill=tk.BOTH)
        
        # Gain Frame
        self.frame1 = tk.Frame(master=self.notebook,bg=self.bg)
        
        self.graph1 = tk.Label(master=self.frame1, image = self.img,bg=self.bg)
        self.graph1.pack(side=tk.TOP)
        
        self.notebook.add(self.frame1,text="Total Gains")
        
        # Gain Frame
        if self.bt.adv_plots == True:
            self.frame2 = tk.Frame(master=self.notebook,bg=self.bg)

            self.graph2 = tk.Label(master=self.frame2, image = self.indimg,bg=self.bg)
            self.graph2.pack(side=tk.TOP)
        
            self.notebook.add(self.frame2,text="L/S Gains")
        
        # Additional Statistics window
        self.addstatbook = ttk.Notebook(self.leftframe)
        self.addstatbook.pack(side=tk.BOTTOM,expand=True,fill=tk.BOTH)
        
        if self.bt.adv_plots == True:
            # Vol Frame
            self.bvolframe = tk.Frame(master=self.addstatbook, bg=self.bg)

            self.bvolgraph = tk.Label(master=self.bvolframe, image = self.bvolimg, bg=self.bg)
            self.bvolgraph.pack(side=tk.TOP)


            self.addstatbook.add(self.bvolframe,text="Buy Volume")

            # Vol Frame
            self.svolframe = tk.Frame(master=self.addstatbook, bg=self.bg)

            self.svolgraph = tk.Label(master=self.svolframe, image = self.svolimg, bg=self.bg)
            self.svolgraph.pack(side=tk.TOP)

            self.addstatbook.add(self.svolframe,text="Sell Volume")
        
        # Trade Window (TOP RIGHT)
        self.statwind = tk.Frame(master=self.rightframe, bg=self.bg,borderwidth=4,relief=tk.RIDGE)

        
        stitle = tk.Label(master=self.statwind, text="Algorithm Statistics",bg=self.bg,borderwidth=8,relief=tk.FLAT,fg='black',font=("Arial", 14))
        
        d1 = tk.Label(self.statwind, text="Start Date: "+str(self.bt.real_start_date),bg=self.bg,borderwidth=4,relief=tk.RIDGE,fg='black',font=("Arial", 13))
        d2 = tk.Label(self.statwind, text="End Date: "+ str(self.bt.end_date),bg=self.bg,borderwidth=4,relief=tk.RIDGE,fg='black',font=("Arial", 13))
        
        l1 = tk.Label(self.statwind, text="Sharpe Ratio: "+self.sharpe_val,bg=self.bg,borderwidth=4,relief=tk.RIDGE,fg='black',font=("Arial", 13))
        l2 = tk.Label(self.statwind, text="Trade Frequency: "+str(round(self.bt.trades/self.bt.num_bdays,2))+ "/day",bg=self.bg,borderwidth=4,relief=tk.RIDGE,fg='black',font=("Arial", 13))
        
        l3 = tk.Label(self.statwind, text="CAG: "+self.compret,bg=self.bg,borderwidth=4,relief=tk.RIDGE,fg='black',font=("Arial", 13))
        l4 = tk.Label(self.statwind, text="Total Return: "+self.totret,bg=self.bg,borderwidth=4,relief=tk.RIDGE,fg='black',font=("Arial", 13))
        
        l5 = tk.Label(self.statwind, text="Win Percentage: "+str(round(self.bt.win/self.bt.trades*100,3))+"%",bg=self.bg,borderwidth=4,relief=tk.RIDGE,fg='black',font=("Arial", 13))
        l6 = tk.Label(self.statwind, text="Average Win: "+str(round(self.bt.win_total/self.bt.win,2))+"%",bg=self.bg,borderwidth=4,relief=tk.RIDGE,fg='black',font=("Arial", 13))
        
        l7 = tk.Label(self.statwind, text="Loss Percentage: "+str(round(self.bt.loss/self.bt.trades*100,3))+"%",bg=self.bg,borderwidth=4,relief=tk.RIDGE,fg='black',font=("Arial", 13))
        l8 = tk.Label(self.statwind, text="Average Loss: "+str(round(self.bt.loss_total/self.bt.loss,2))+"%",bg=self.bg,borderwidth=4,relief=tk.RIDGE,fg='black',font=("Arial", 13))
        
        l9 = tk.Label(self.statwind, text="Trade Expected Value: "+str(round((self.bt.win_total-self.bt.loss_total)/self.bt.trades,2))+"%",bg=self.bg,borderwidth=4,relief=tk.RIDGE,fg='black',font=("Arial", 13))
        l10 = tk.Label(self.statwind, text="Volatility: "+str(round(self.bt.calc_vol(),2))+"%",bg=self.bg,borderwidth=4,relief=tk.RIDGE,fg='black',font=("Arial", 13))
        
        stitle.grid(row = 0, column = 0, sticky = tk.EW, columnspan=2)
        
        d1.grid(row = 1, column = 0, sticky = tk.EW)
        d2.grid(row = 1, column = 1, sticky = tk.EW)
        
        l1.grid(row = 2, column = 0, sticky = tk.EW)
        l2.grid(row = 2, column = 1, sticky = tk.EW)
        
        l3.grid(row = 3, column = 0, sticky = tk.EW)
        l4.grid(row = 3, column = 1, sticky = tk.EW)
        
        l5.grid(row = 4, column = 0, sticky = tk.EW)
        l6.grid(row = 4, column = 1, sticky = tk.EW)
        
        l7.grid(row = 5, column = 0, sticky = tk.EW)
        l8.grid(row = 5, column = 1, sticky = tk.EW)
        
        l9.grid(row = 6, column = 0, sticky = tk.EW, columnspan=2)
        l10.grid(row = 7, column = 0, sticky = tk.EW, columnspan=2)
        
        self.statwind.grid_columnconfigure(0,weight=1)
        self.statwind.pack(side=tk.TOP,fill=tk.X,expand=False)
        
        # Portfolio Window  (BOTTOM RIGHT)
        self.logwind = tk.Frame(master=self.rightframe, bg=self.bg,borderwidth=4,relief=tk.RIDGE)
        
        ltitle = tk.Label(master=self.logwind, text="Trading Log",bg=self.bg,borderwidth=4,relief=tk.FLAT,fg='black',font=("Arial", 14))
        ltitle.pack(side=tk.TOP)
        
        logbox = scrolledtext.ScrolledText(self.logwind, undo=True,borderwidth=4,relief=tk.RIDGE)
        logbox['font'] = ('consolas', '14')
        logbox.insert(tk.END, self.bt.log)
        logbox.pack(side=tk.TOP,fill=tk.BOTH,expand=True)
        
        self.logwind.pack(side=tk.BOTTOM,fill=tk.BOTH,expand=True)                                             

    def display(self):
        """
        Shows display.
        """
        self.window.mainloop()
    
    def refresh(self):
        """
        Refeshes Display. To be used for trading.
        """
        pass