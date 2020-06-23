import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import df_between



class Strategy:
    def __init__(self, model, take_profit_coef, stop_loss_coef, broker_comission, close_interval):
        self.model = model
        self.take_profit_coef = take_profit_coef#1 + 0.005
        self.stop_loss_coef = stop_loss_coef#1 - 0.005
        self.broker_comission = broker_comission
        self.close_interval = close_interval
        
        self.source_pred = None
        self.orders = None
        
        
    def evaluate(self, ticker2df, X_val, capital=100_000):
        self.create_orders(ticker2df, X_val)
        self.orders['abs_profit'] = self.orders['profit'] * capital
        self.orders['time_in_order'] = self.orders['sell_datetime'] - self.orders['buy_datetime']
        self.orders['date'] = self.orders['buy_datetime'].astype('datetime64[D]')
        
        self.grouped_orders = self.orders.groupby(['ticker','date'])['abs_profit'].sum().reset_index()
        
        return self.orders['abs_profit'].sum()
        
    
    def create_orders(self, ticker2df, X_val):        
#         pred = self.model.predict_proba(X_val.drop(['corn_datetime', 'ticker'], axis=1))[:, 1]
        
#         buy_moments = X_val[pred > 0.5][['ticker', 'corn_datetime']]

        buy_moments, source_pred = self.model.predict_moments(X_val)
        self.source_pred = source_pred
        
        orders = []
        sell_datetime = None
        for moment in buy_moments.itertuples():
            ticker = moment.ticker
            buy_datetime = moment.corn_datetime
            
            if (sell_datetime is not None) and buy_datetime <= sell_datetime:
                continue
        
            target_df = df_between(ticker2df[ticker], buy_datetime, buy_datetime + self.close_interval)
            target_df.index = range(len(target_df))
            series = target_df['<OPEN>'].values

            take_profit_price = series[0] * self.take_profit_coef
            stop_loss_price = series[0] * self.stop_loss_coef

            take_profit_mask = series > take_profit_price
            stop_loss_mask = series < stop_loss_price

            can_sell = (take_profit_mask | stop_loss_mask).max()

            if can_sell:
                sell_idx = np.where(take_profit_mask | stop_loss_mask)[0][0]
            else:
                sell_idx = len(target_df) - 1

            sell_datetime = target_df.loc[sell_idx]['datetime']
            profit = (series[sell_idx] - series[0]) / series[0] - self.broker_comission

            orders.append({'ticker':ticker, 'buy_datetime':buy_datetime, 'sell_datetime':sell_datetime,\
                           'profit':profit, 'buy_price':series[0], 'sell_price':series[sell_idx]})
                
        self.orders = pd.DataFrame(orders)
        
        
    def draw(self, ticker2df, ticker_date_k):
        ticker_date = self.grouped_orders.loc[ticker_date_k]
        ticker = ticker_date['ticker']
        date = ticker_date['date']
        
        day_orders = self.orders[(self.orders['date'] == date) & (self.orders['ticker']==ticker)]
        
        day_price = ticker2df[ticker][ticker2df[ticker]['date'] == date]#['<OPEN>']
        #day_price['norm_price'] = (day_price['<OPEN>'] - day_price['<OPEN>'].mean()) / day_price['<OPEN>'].std()
        
        day_source = self.source_pred[(self.source_pred['date']==date) & (self.source_pred['ticker']==ticker)]
                
        
        fig=plt.figure(figsize=(25,10))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212, sharex = ax1)
        plt.title(ticker)
        for order in day_orders.itertuples():
            ax1.scatter(order.buy_datetime, order.buy_price, color='g')
            ax1.scatter(order.sell_datetime, order.sell_price, color='r')
            
        ax1.plot(day_price['datetime'].values, day_price['<OPEN>'])
        ax2.plot(day_source['datetime'].values, day_source['pred'].values)

        
class Ansamble:
    def __init__(self):
        self.models = []
        

    def fit(self, X, y_series):
            
        for thr in [0.01, 0.015]:
            y_train = calc_target(y_series, single_target_3) > thr
            lgb = lgbm.sklearn.LGBMClassifier()
            self.models.append(lgb.fit(X.drop(['corn_date', 'ticker'], axis=1), y_train))    

        for thr in [0.0, 0.01, 0.015]:
            y_train = calc_target(y_series, single_target_2) > thr
            lgb = lgbm.sklearn.LGBMClassifier()
            self.models.append(lgb.fit(X.drop(['corn_date', 'ticker'], axis=1), y_train))    

        for thr in [0.0, 0.01, 0.015]:
            y_train = calc_target(y_series, single_target_4) > thr
            lgb = lgbm.sklearn.LGBMClassifier()
            self.models.append(lgb.fit(X.drop(['corn_date', 'ticker'], axis=1), y_train)) 

        for thr in [0.0, 0.01, 0.015]:
            y_train = calc_target(y_series, single_target_5) > thr
            lgb = lgbm.sklearn.LGBMClassifier()
            self.models.append(lgb.fit(X.drop(['corn_date', 'ticker'], axis=1), y_train)) 

        
    def predict_proba_matrix(self, X):
        all_probas = []
        for model in self.models:
            pred_proba = model.predict_proba(X.drop(['corn_date', 'ticker'], axis=1))[:, 1]
            all_probas.append(pred_proba)
            
        all_probas = np.array(all_probas)
        
        return all_probas
        
    def predict_proba(self, X):
        proba_matrix = self.predict_proba_matrix(X)
        proba = proba_matrix.sum()
        
        return proba
    
# ansamble = Ansamble()
# ansamble.fit(X_train, y_train_series)
# proba_matrix = ansamble.predict_proba_matrix(X_val)

