import argparse
import datetime
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
import pandas as pd
import numpy as np
from utils import round_price, process_cnt
from parameters import ibkr_info


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
            print("Executing requests")
            #self.historicalDataOperations_req()
            self.my_order_req()
            #self.my_order_req()




    def nextOrderId(self):
        oid = self.nextValidOrderId
        self.nextValidOrderId += 1
        return oid



    def my_order_req(self):

        self.reqIds(-1)



        #ticker2price = {'GAZP': 223.62, 'NVTK': 1289.6, 'YNDX': 2380.0}
        ticker2price = {'FEES': 0.1806, 'GMKN': 15040.0, 'POLY': 836.9, 'ROSN': 407.5, 'RUAL': 26.995}
        ticker2price = {'PLZL': 7187.5, 'RUAL': 26.55}
        ticker2price = {'AFKS': 11.56, 'ALRS': 73.01, 'MAGN': 38.945, 'PLZL': 6958.5, 'YNDX': 2440.0}
        ticker2price = {'AFKS': 11.141, 'PLZL': 7222.0, 'VTBR': 0.039235}
        ticker2price = {'VTBR': 0.039235}
        ticker2price = {'MAGN': 38.4, 'TRNFP': 151850.0}
        ticker2price = {'AFKS': 11.087, 'RUAL': 25.9, 'TRMK': 52.98}
        
        for ticker in ticker2price:
            price = ticker2price[ticker]
                        
            cnt = 2000000 / (len(ticker2price)) / price
            cnt = (cnt // ibkr_info[ticker]['min_quantity']) * ibkr_info[ticker]['min_quantity']


            start_price = round_price(price*1.003)
            profit_price = round_price(price*1.01)

        
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
