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

from ContractSamples import ContractSamples
from OrderSamples import OrderSamples
from AvailableAlgoParams import AvailableAlgoParams
from ScannerSubscriptionSamples import ScannerSubscriptionSamples
from FaAllocationSamples import FaAllocationSamples
from ibapi.scanner import ScanData
from ibapi import utils
import pandas as pd
import numpy as np


# ! [socket_declare]
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



    def process_cnt(self, cnt):
        if cnt > 50:
            cnt = cnt // 10 * 10

        if cnt > 500:
            cnt = cnt // 100 * 100

        return cnt


    def round_price(self, price):
        ready = False
        for k in range(1, 10):
            new_price = round(price, k)
            if np.abs(price - new_price) / price < 0.0001:
                last_ch_new_price = float(str(new_price)[:-1] + str(int(str(new_price)[-1]) // 2 * 2))
                if np.abs(price - last_ch_new_price) / price < 0.0001:
                    return last_ch_new_price
                else:
                    return new_price

    def my_order_req(self):

        self.reqIds(-1)



        #ticker2price = {'GAZP': 223.62, 'NVTK': 1289.6, 'YNDX': 2380.0}
        ticker2price = {'MGNT': 3600.0, 'TRMK': 56.56}

        for ticker in ticker2price:
            ticker = 'MGNT'
            price = ticker2price[ticker]
            cnt = 400000 / (len(ticker2price)) / price
            cnt = self.process_cnt(cnt)
            start_price = self.round_price(price*1.003)
            profit_price = self.round_price(price*1.01)

            contract = ContractSamples.MYStock(ticker)

            order_id = self.nextOrderId()
            self.nextOrderId()
            print("order_id: {}, cnt: {}, start_price: {}, profit_price: {}".format(order_id, cnt, start_price, profit_price))


            parent, takeProfit = OrderSamples.TopBracketOrder(order_id, "BUY", cnt, 3560, profit_price)
            self.placeOrder(parent.orderId, contract, parent)
            self.placeOrder(takeProfit.orderId, contract, takeProfit)

            break



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
