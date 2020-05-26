from urllib.parse import urlencode
from urllib.request import urlopen
from datetime import datetime
from parameters import ticker2finam
from multiprocessing import Pool
from tqdm import tqdm
from itertools import repeat
from datetime import timedelta
import numpy as np
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from tqdm import tqdm
import os
import pandas as pd
import itertools
import time


def load_single(ticker, data_prefix, start_date, end_date, period=3):        
    FINAM_URL = "http://export.finam.ru/"# сервер, на который стучимся
    market = 0 #можно не задавать. Это рынок, на котором торгуется бумага. Для акций работает с любой цифрой. Другие рынки не проверял.
    #Делаем преобразования дат:
    #start_date = datetime.strptime(start, "%d.%m.%Y").date()
    start_date_rev=start_date.strftime('%Y%m%d')
    #end_date = datetime.strptime(end, "%d.%m.%Y").date()
    end_date_rev=end_date.strftime('%Y%m%d')


    #Все параметры упаковываем в единую структуру. Здесь есть дополнительные параметры, кроме тех, которые заданы в шапке. См. комментарии внизу:
    params = urlencode([
                        ('market', market), #на каком рынке торгуется бумага
                        ('em', ticker2finam[ticker]), #вытягиваем цифровой символ, который соответствует бумаге.
                        ('code', ticker), #тикер нашей акции
                        ('apply',0), #не нашёл что это значит. 
                        ('df', start_date.day), #Начальная дата, номер дня (1-31)
                        ('mf', start_date.month - 1), #Начальная дата, номер месяца (0-11)
                        ('yf', start_date.year), #Начальная дата, год
                        ('from', start_date), #Начальная дата полностью
                        ('dt', end_date.day), #Конечная дата, номер дня	
                        ('mt', end_date.month - 1), #Конечная дата, номер месяца
                        ('yt', end_date.year), #Конечная дата, год
                        ('to', end_date), #Конечная дата
                        ('p', period), #Таймфрейм
                        ('f', ticker+"_" + start_date_rev + "_" + end_date_rev), #Имя сформированного файла
                        ('e', ".csv"), #Расширение сформированного файла
                        ('cn', ticker), #ещё раз тикер акции	
                        ('dtf', 1), #В каком формате брать даты. Выбор из 5 возможных. См. страницу https://www.finam.ru/profile/moex-akcii/sberbank/export/
                        ('tmf', 1), #В каком формате брать время. Выбор из 4 возможных.
                        ('MSOR', 0), #Время свечи (0 - open; 1 - close)	
                        ('mstime', "on"), #Московское время	
                        ('mstimever', 1), #Коррекция часового пояса	
                        ('sep', 1), #Разделитель полей	(1 - запятая, 2 - точка, 3 - точка с запятой, 4 - табуляция, 5 - пробел)
                        ('sep2', 1), #Разделитель разрядов
                        ('datf', 1), #Формат записи в файл. Выбор из 6 возможных.
                        ('at', 1)]) #Нужны ли заголовки столбцов
    url = FINAM_URL + ticker+"_" + start_date_rev + "_" + end_date_rev + ".csv?" + params #урл составлен!
    txt=urlopen(url).readlines() #здесь лежит огромный массив данных, прилетевший с Финама.
    local_file = open('{}/{}.csv'.format(data_prefix, ticker), "w") #задаём файл, в который запишем котировки.
    for line in txt: #записываем свечи строку за строкой. 
        local_file.write(line.strip().decode( "utf-8" )+'\n')
    local_file.close()


def load_tickers(data_prefix, tickers, start_date, end_date, period=3):
    for ticker in tqdm(tickers):
        try:
            load_single(ticker, data_prefix, start_date, end_date, period)
            time.sleep(0.6)
        except:
            print(ticker)
    
    
def load_dfs(dir_path, tickers):
    ticker2df = {}
    for ticker in tickers:
        #try:
            df = pd.read_csv('{}/{}.csv'.format(dir_path, ticker))
            #df['date'] = df['<DATE>'].apply(lambda x: datetime.strptime(str(x)+, "%Y%m%d%H").date())
            df['date'] = df.apply(lambda x: datetime.strptime(str(x['<DATE>']), "%Y%m%d"), axis=1)
            df['datetime'] = df.apply(lambda x: datetime.strptime(str(x['<DATE>'])+str(x['<TIME>'])[:4], "%Y%m%d%H%M"), axis=1)

            if df.shape[0] < 300:
                continue

            ticker2df[ticker] = df
#         except:
#             print(ticker)

    return ticker2df




def single_target_1(series):
    profit = (series[-1] - series[1]) / series[0]
    return profit

def single_target_2(series):
    profit = (series[-1] - series[0]) / series[0]
    return profit

def single_target_3(series):
    profit = (series[1:].max() - series[0]) / series[0]
    return profit

def single_target_4(series):
    profit = (series[1:].min() - series[0]) / series[0]
    return profit

def single_target_5(series):
    profit = (series[1:].mean() - series[0]) / series[0]
    return profit

def single_target_6(series):
    profit = series[1:].std() / series[0]
    return profit

def single_target_7(series):
    profit = series[1:].max() / series[1:].min()
    return profit


def target_df_2_one_day_series(target_df):
    return target_df[target_df['date'] == target_df['corn_date']]['<OPEN>'].values

def target_df_2_free_day_series(target_df):
    return df_future(target_df[target_df['date'] >= target_df['corn_date']], 3)['<OPEN>'].values
    
def target_df_2_half_day_series(target_df):
    one_day_series = df_future(target_df[target_df['date'] >= target_df['corn_date']], 3)['<OPEN>'].values
    return one_day_series[:len(one_day_series) // 2]
    


def single_profit_2(series, return_idxs=False, TAKE_PROFIT_COEF = 1.01, STOP_LOSS_COEF = 0.99):

    #profit = (series[-5:-4].min() - series[1]) / series[1]#  > 0.01
    #profit = (series[-1] - series[1]) / series[1]  #> 0.01

    UPPER_COEF = 1.001
    BUY_HORIZON = 8
    

    
    can_buy = (series[1:BUY_HORIZON] <= series[0] * UPPER_COEF).max()
    if can_buy:
        buy_idx = np.where(series[0:BUY_HORIZON] <= series[0]*UPPER_COEF)[0][0] + 1

        take_profit_price = series[buy_idx] * TAKE_PROFIT_COEF
        stop_loss_price = series[buy_idx] * STOP_LOSS_COEF

        take_profit_mask = series[buy_idx:-3] > take_profit_price
        stop_loss_mask = series[buy_idx:-3] < stop_loss_price

        can_sell = (take_profit_mask | stop_loss_mask).max()
        #can_sell = (take_profit_mask ).max()

        if can_sell:
            #sell_idx = np.where(take_profit_mask )[0][0] + 2
            sell_idx = np.where(take_profit_mask | stop_loss_mask)[0][0] + buy_idx
            
            profit = (series[sell_idx] - series[buy_idx]) / series[buy_idx]
            profit = min(0.012, profit)
        else:
            sell_idx = len(series)-3

            profit = (series[sell_idx] - series[buy_idx]) / series[buy_idx]
  
    else:
        profit = 0
        buy_idx = len(series)-1
        sell_idx = len(series)-1
        
    if return_idxs:
        return buy_idx, sell_idx, profit
    else:
        return profit



def single_profit(series):
    if len(series) > 10:
        UPPER_COEF = 1.003
        can_buy = (series[2:6] <= series[1]*UPPER_COEF).max()
        if can_buy:
            buy_idx = np.where(series[2:6] <= series[1]*UPPER_COEF)[0][0] + 2
            sell_idx = len(series[:-5]) + np.random.randint(5)#+ series[-5:].argmin()
            sell_idx = -1
            profit = (series[sell_idx] - series[buy_idx]) / series[buy_idx]
        else:
            profit = 0
    else:
        profit = 0
        
    return profit


def calc_target(target_dfs, foo, df2series_foo):
    target = []
    for ticker_targets in target_dfs:
        for target_df in ticker_targets:
            val = foo(df2series_foo(target_df))
            target.append(val)
            
    return np.array(target)


def split_train_target(df, corn_datetime, target_interval):
    train_df = df_between(df, None, corn_datetime)
    train_df['corn_datetime'] = corn_datetime
    
    target_df = df_between(df, corn_datetime, corn_datetime + target_interval)
    target_df['corn_datetime'] = corn_datetime
    
    return train_df, target_df

    
def single_ticker(ticker_df, datetimes, target_interval):
    ticker_feats, ticker_targets = [], []
    
    for corn_datetime in datetimes:
        train_df, target_df = split_train_target(ticker_df, corn_datetime, target_interval)
        feats = calc_feat(train_df)
        target = single_target_6(target_df['<OPEN>'].values)
        ticker_feats.append(feats)
        ticker_targets.append(target)
        
    if len(ticker_feats) > 0:
        ticker_feats = pd.concat(ticker_feats, axis=0)            
            
    return ticker_feats, ticker_targets


def all_samples(ticker2df, datetimes, target_interval):
    p = Pool(20)
    feats_res = []
    y_res = []
    res = p.starmap(single_ticker, zip(ticker2df.values(), repeat(datetimes), repeat(target_interval)))

    #return res
    for feat, y in res:
        if len(feat) > 0:
            feats_res.append(feat)
            y_res.append(y)
         
    feats_res = pd.concat(feats_res, axis=0)
    y_res = np.concatenate(y_res, axis=0)

    return feats_res, y_res    




def df_between(df, start_date=None, end_date=None):
    mask = np.ones(len(df), dtype=bool)
    if start_date is not None:
        mask_start = df['datetime'].apply(lambda x: x >= start_date).values
        mask = mask * mask_start

    if end_date is not None:
        mask_end = df['datetime'].apply(lambda x: x < end_date).values
        mask = mask * mask_end

    return df.loc[mask]


def df_last(df, day_cnt):
    dates = np.sort(df['date'].unique())[-day_cnt:]
    start_date = dates[0]
    end_date = dates[-1] + np.timedelta64(1,'D')
    return df_between(df, start_date, end_date)


def df_future(df, day_cnt):
    dates = np.sort(df['date'].unique())[:day_cnt]
    start_date = dates[0]
    end_date = dates[-1] + timedelta(days=1)
    return df_between(df, start_date, end_date)


def calc_base_time_feats(series):
#     if len(series) == 0:
#         series = np.array([np.nan])
        
    feats = [series.std() / series[0],
             series.mean() / series[0],
             series.min() / series[0],
             np.median(series) / series[0],
             (series.max() - series.min()) / series[0],
             (series[-1] - series[0]) / series[0]
            ]
    
    
    return feats


def calc_support_feats(series):
    if len(series) == 0:
        series = np.array([np.nan])
        
    support_levels = calc_support_levels(series)
    
    if len(support_levels) == 0:
        support_levels = np.array([np.nan])
        
    
    support_diff = series[-1] - support_levels

    upper_supports = support_levels[support_diff < 0]
    lower_supports = support_levels[support_diff > 0]
    
    if len(upper_supports) == 0:
        upper_supports = np.array([np.nan])        
        
    if len(lower_supports) == 0:
        lower_supports = np.array([np.nan])        
            
    if len(series) == 0:
        series = np.array([np.nan])
        
        
    upper_supports_diff = np.abs(upper_supports - series[-1])
    lower_supports_diff = np.abs(lower_supports - series[-1])
   
    support_features = [len(upper_supports[~np.isnan(upper_supports)]),
                        len(lower_supports[~np.isnan(lower_supports)]),
                        upper_supports.min() / series.max(),
                        lower_supports.max() / series.max(),
                        support_levels.max() / series.max(),
                        support_levels.min() / series.max(),
                        upper_supports_diff.min() / series.max(),
                        lower_supports_diff.min() / series.max()]
    
    
    return support_features
   
def base_aggs(series):
    aggs = np.array([series.mean(), series.max(), series.min(), np.median(series)])
    return aggs
    
def calc_target_like_feats(df):
    groupby_date = df.groupby('date')['<OPEN>']
    target_aggs_1 = base_aggs(groupby_date.apply(lambda x: single_target_1(np.array(x.tolist()))).values)
    target_aggs_2 = base_aggs(groupby_date.apply(lambda x: single_target_2(np.array(x.tolist()))).values)
    target_aggs_3 = base_aggs(groupby_date.apply(lambda x: single_target_3(np.array(x.tolist()))).values)
    target_aggs_4 = base_aggs(groupby_date.apply(lambda x: single_target_4(np.array(x.tolist()))).values)
    target_aggs_5 = base_aggs(groupby_date.apply(lambda x: single_target_5(np.array(x.tolist()))).values)
    target_aggs_6 = base_aggs(groupby_date.apply(lambda x: single_target_6(np.array(x.tolist()))).values)
    target_aggs_7 = base_aggs(groupby_date.apply(lambda x: single_target_7(np.array(x.tolist()))).values)

    target_like_features = np.concatenate([target_aggs_1, target_aggs_2, target_aggs_3, target_aggs_4, target_aggs_5, target_aggs_6, target_aggs_7], axis=0)    

    return target_like_features

    
def calc_night_gaps_feats(df):
    open_prices = df.groupby('date')['<OPEN>'].apply(lambda x:x.tolist()[0]).reset_index()['<OPEN>'].values
    close_prices = df.groupby('date')['<CLOSE>'].apply(lambda x:x.tolist()[-1]).reset_index()['<CLOSE>'].values

    night_gaps = (open_prices[1:] - close_prices[:-1]) / close_prices[:-1]
    
    return night_gaps.mean(), night_gaps.min(), night_gaps.max(), night_gaps.std(), np.median(night_gaps), np.abs(night_gaps).mean()


def calc_base_feats_aggs_by_day(df):
    base_feats_by_day = np.array([x for x in df.groupby('date')['<OPEN>'].apply(lambda x: calc_base_time_feats(np.array(x.tolist()))).values])
    base_feats_aggs_by_day = np.concatenate([base_feats_by_day.mean(axis=0),
                                             base_feats_by_day.max(axis=0),
                                             base_feats_by_day.min(axis=0),
                                             base_feats_by_day.std(axis=0),
                                             np.median(base_feats_by_day, axis=0)])

    return base_feats_aggs_by_day


def calc_next_day_feats(df, target_df):
    corn_date = target_df['corn_date'].values[0]
    corn_date_df = target_df[target_df['date'] == corn_date]
    
    last_day_df = df_between(df, start_date=None, end_date=corn_date)
    last_day_df = df_last(last_day_df, 1)
    
    if len(corn_date_df) > 0:
        open_price = corn_date_df['<OPEN>'].values[0]
        close_price = last_day_df['<CLOSE>'].values[-1]
        next_day_feat = [(open_price - close_price) / close_price]
    else:
        next_day_feat = [np.nan]
        
    return next_day_feat
    
    
def calc_feat(df):
    # Support values
    corn_datetime = df['corn_datetime'].values[0]
    ticker = df['<TICKER>'].values[0] 
    
    month_df = df_between(df, corn_datetime - np.timedelta64(31,'D'), corn_datetime)
    week_df = df_between(df, corn_datetime - np.timedelta64(7,'D'), corn_datetime)
    
    month_series = month_df['<OPEN>'].values
    week_series = week_df['<OPEN>'].values
    
    day3_df = df_between(df, start_date=None, end_date=corn_datetime)
    day3_df = df_last(day3_df, 3)
    day3_series = day3_df['<OPEN>'].values
    
    # Calculate features
    month_feats = calc_base_time_feats(month_series)
    week_feats = calc_base_time_feats(week_series)
    day3_feats = calc_base_time_feats(day3_series)    
    support_feats = calc_support_feats(month_series)
    #target_like_feats = calc_target_like_feats(df_between(df, corn_date - timedelta(days=31), corn_date))
    months_base_feats_aggs_by_day = calc_base_feats_aggs_by_day(month_df)
    week_base_feats_aggs_by_day = calc_base_feats_aggs_by_day(week_df)
    month_night_gaps_feats = calc_night_gaps_feats(month_df)
    week_night_gaps_feats = calc_night_gaps_feats(week_df)
    
    result = np.concatenate([
                             month_feats,
                             week_feats,
                             day3_feats,
                             support_feats,
                             #target_like_feats,
                             months_base_feats_aggs_by_day,
                             week_base_feats_aggs_by_day,
                             month_night_gaps_feats,
                             week_night_gaps_feats
                            ], axis=0)
    

    # Form result
    result = np.expand_dims(result, axis=0)
    result = pd.DataFrame(result)
    
    result['ticker'] = ticker
    result['corn_datetime'] = corn_datetime
    
    return result


def get_month_series(df, corn_date):
    month_series = df_between(df, corn_date - timedelta(days=31), corn_date)['<OPEN>'].values
    return month_series
    

        
def calc_support_levels(series):
    if len(series) < 2:
        return []
    line_cnts = []
    line_vals = np.linspace(series.min(), series.max(), 100)
    for line in line_vals:
        line_cnts.append((np.abs(series - line) / series.max() < 0.001).sum())

    sm_line_cnts = savgol_filter(line_cnts, 11, 3)
    loc_maximum_idxs = argrelextrema(sm_line_cnts, np.greater)[0]
    loc_maximums = line_vals[loc_maximum_idxs]
    
    return loc_maximums





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

# sns.heatmap(pd.DataFrame(proba_matrix.T).corr())

# plt.plot(proba_matrix.sum(axis=0))    



        
        
        