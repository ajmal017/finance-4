import argparse
from datetime import datetime
import collections
import inspect
import logging
import time
import os.path

from ibapi import wrapper
from ibapi import utils
from ibapi.client import EClient
from ibapi.utils import iswrapper

# types
from ibapi.common import * # @UnusedWildImport
from ibapi.order_condition import * # @UnusedWildImport
from ibapi.contract import * # @UnusedWildImport
from ibapi.order import * # @UnusedWildImport
from ibapi.order_state import * # @UnusedWildImport
from ibapi.execution import Execution
from ibapi.execution import ExecutionFilter
from ibapi.commission_report import CommissionReport
from ibapi.ticktype import * # @UnusedWildImport
from ibapi.tag_value import TagValue

from ibapi.account_summary_tags import *

from ibkr_api.ContractSamples import ContractSamples
from ibkr_api.OrderSamples import OrderSamples
from ibkr_api.AvailableAlgoParams import AvailableAlgoParams
from ibkr_api.ScannerSubscriptionSamples import ScannerSubscriptionSamples
from ibkr_api.FaAllocationSamples import FaAllocationSamples
from ibapi.scanner import ScanData
from ibapi import utils
#import pandas as pd
import numpy as np
from utils import load_tickers, load_dfs, all_samples
from parameters import ibkr_info
import pickle

from datetime import datetime
from datetime import timedelta


class TestClient(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)


class TestWrapper(wrapper.EWrapper):
    def __init__(self):
        wrapper.EWrapper.__init__(self)

        self.wrapMeth2callCount = collections.defaultdict(int)
        self.wrapMeth2reqIdIdx = collections.defaultdict(lambda: -1)
        self.reqId2nAns = collections.defaultdict(int)


class TestApp(TestWrapper, TestClient):
    def __init__(self):
        TestWrapper.__init__(self)
        TestClient.__init__(self, wrapper=self)
        self.started = False
        self.nextValidOrderId = None
        self.globalCancelOnly = False
        self.simplePlaceOid = None
        self.data = []
        #self.


    @iswrapper
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId
        self.start()

    def start(self):
        if self.started:
            return

        self.started = True

        if self.globalCancelOnly:
            print("Executing GlobalCancel only")
            self.reqGlobalCancel()
        else:
            with open('models/lgb_26_09_thr04.pickle', 'rb') as f:
                lgb = pickle.load(f)
            
            while int(datetime.now().strftime('%H')) != 10:
                #print(int(datetime.now().strftime('%H')))
                time.sleep(0.05)
                continue            
            
            
            print('Loading tickers')
            
            load_tickers(data_prefix="data/current", tickers=np.array(list(ibkr_info.keys())), start_date=datetime.today().date() - timedelta(days=40), end_date=datetime.today().date(), period=3)

            ticker2df_test = load_dfs('data/current', np.array(list(ibkr_info.keys())))
            
            X_test, y_test_series = all_samples(ticker2df_test, [datetime.today().date()], test_mode=True)
            
            #self.historicalDataOperations_req()
            
            
            pred_proba = lgb.predict_proba(X_test.drop(['corn_date', 'ticker'], axis=1))[:, 1]

            print(pred_proba)
            #print("Start order commiting")
            top_idxs = np.where(pred_proba > 0.35)[0]
            ticker2price = {}

            for ticker in list(X_test.loc[top_idxs, ['ticker', 'corn_date']]['ticker'].values):

                df = ticker2df_test[ticker]
                price = df[df['date']==datetime.today().date()]['<OPEN>'].values[0]

                ticker2price[ticker] = price
            
            
            self.my_order_req(ticker2price)
            #self.my_order_req()




    def nextOrderId(self):
        oid = self.nextValidOrderId
        self.nextValidOrderId += 1
        return oid



    def my_order_req(self, ticker2price):
        self.reqIds(-1)

        print(ticker2price)
        
        for ticker in ticker2price:
            price = ticker2price[ticker]
                        
            cnt = 2300000 / (len(ticker2price)) / price
            cnt = (cnt // ibkr_info[ticker]['min_quantity']) * ibkr_info[ticker]['min_quantity']


            start_price = price*1.003
            profit_price = price*1.01

        
            start_price = (start_price // ibkr_info[ticker]['precise']) * ibkr_info[ticker]['precise']
            profit_price = (profit_price // ibkr_info[ticker]['precise']) * ibkr_info[ticker]['precise']
        
        
            contract = ContractSamples.MYStock(ticker)

            order_id = self.nextOrderId()
            self.nextOrderId()
            print("order_id: {}, cnt: {}, start_price: {}, profit_price: {}".format(order_id, cnt, start_price, profit_price))


            parent, takeProfit = OrderSamples.TopBracketOrder(order_id, "BUY", cnt, start_price, profit_price)
            self.placeOrder(parent.orderId, contract, parent)
            self.placeOrder(takeProfit.orderId, contract, takeProfit)

            



def main():
    cmdLineParser = argparse.ArgumentParser("api tests")
    cmdLineParser.add_argument("-p", "--port", action="store", type=int,
                               dest="port", default=7497, help="The TCP port to use")
    cmdLineParser.add_argument("-C", "--global-cancel", action="store_true",
                               dest="global_cancel", default=False,
                               help="whether to trigger a globalCancel req")
    args = cmdLineParser.parse_args()
    print("Using args", args)




    app = TestApp()

    app.connect("127.0.0.1", args.port, clientId=0)
    print("serverVersion:%s connectionTime:%s" % (app.serverVersion(),
                                                      app.twsConnectionTime()))


    app.run()



if __name__ == "__main__":
    main()
