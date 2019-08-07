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
            self.historicalDataOperations_req()
            #self.my_order_req()




    def nextOrderId(self):
        oid = self.nextValidOrderId
        self.nextValidOrderId += 1
        return oid




    def historicalDataOperations_req(self):
        queryTime = (datetime.datetime.today() - datetime.timedelta(days=180)).strftime("%Y%m%d %H:%M:%S")

        res = self.reqHistoricalData(0, ContractSamples.EurGbpFx(), "",
                               "1 Y", "15 mins", "MIDPOINT", 1, 1, True, [])
        

    @iswrapper
    def historicalData(self, reqId:int, bar: BarData):
        self.data.append({"date": bar.date,
                          "open": bar.open,
                          "close": bar.close,
                          "low": bar.low,
                          "high": bar.high})



    @iswrapper
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        super().historicalDataEnd(reqId, start, end)
        #import pickle
        #with open('data.pickle', 'wb') as f:
        #    pickle.dump(self.data, f)
        df = pd.DataFrame(self.data)
        df.to_csv("df.csv", index=False)
        print("Длина скаченной последовательности: {}".format(len(self.data)))



    @iswrapper
    def historicalDataUpdate(self, reqId: int, bar: BarData):
        print("HistoricalDataUpdate. ReqId:", reqId, "BarData.", bar)




    def my_order_req(self):

        self.reqIds(-1)

        self.placeOrder(self.nextOrderId(), ContractSamples.EurGbpFx(),
                        OrderSamples.LimitOrder("BUY", 1, 50))




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
