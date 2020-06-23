from multiprocessing import Pool
from tqdm import tqdm
from itertools import repeat
#import datetime
import numpy as np
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from tqdm import tqdm
import pandas as pd
import time

from utils import split_train_target, df_between, df_last


def target_0(series):
    profit = (series[-1] - series[0]) / series[0]
    return profit

def target_1(series):
    profit = (series.max() - series[0]) / series[0]
    return profit

def target_2(series):
    profit = (series.min() - series[0]) / series[0]
    return profit

def target_3(series):
    profit = (series.mean() - series[0]) / series[0]
    return profit

def target_4(series):
    profit = series.std() / series[0]
    return profit

def target_5(series):
    profit = series.max() / series[1:].min()
    return profit

def target_6(series):
    down_vals = series[series < series.mean()]
    profit = ((down_vals - series[0]) ** 2).sum() ** (1/2) / series[0]
    return profit

def target_7(series):
    up_vals = series[series > series.mean()]
    profit = ((up_vals - series[0]) ** 2).sum() ** (1/2) / series[0]
    return profit

def target_8(series):
    down_vals = series[series < series.mean()]
    profit = ((down_vals - series.mean()) ** 2).sum() ** (1/2) / series[0]
    return profit

def target_9(series):
    up_vals = series[series > series.mean()]
    profit = ((up_vals - series.mean()) ** 2).sum() ** (1/2) / series[0]
    return profit


BASE_FOOS_ARR = [
                 np.mean,
                 np.max,
                 np.std,
                 np.median,
                 target_0,
                 target_1,
                 target_2,
                 target_3,
                 target_4,
                 target_5,
                 target_6,
                 target_7,
                 target_8,
                 target_9,
               ]
    

def precompute_rollings(ticker2df, win_lens=[10,20]):
    for ticker in tqdm(ticker2df.keys()):
        for foo in BASE_FOOS_ARR:
            for win_len in win_lens:
                ticker2df[ticker]['{}_win{}'.format(foo.__name__, win_len)] = ticker2df[ticker].groupby('date')['<OPEN>'].rolling(win_len).apply(lambda x: foo(np.array(list(x)))).shift(-win_len+1).reset_index()['<OPEN>']

                
def calc_introday_rolling_aggs(df):
    rolling_cols = [x for x in df.columns if 'win' in x]
    rolling_source_df = df[rolling_cols]
    feats = rolling_source_df.min().tolist() +\
            rolling_source_df.max().tolist() +\
            rolling_source_df.std().tolist()
    
    return feats

    
def calc_base_time_feats(series):
    series = np.array(series)
    if len(series) < 2:
        feats = [None] * len(BASE_FOOS_ARR)
    else:
        feats = [foo(series) for foo in BASE_FOOS_ARR]
    return feats

        
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

    
    
def calc_feats(df):
    # Support values
    corn_datetime = df['corn_datetime'].values[0]
    ticker = df['<TICKER>'].values[0] 
    
    m10_df = df_between(df, corn_datetime - np.timedelta64(10,'m'), corn_datetime)
    m30_df = df_between(df, corn_datetime - np.timedelta64(30,'m'), corn_datetime)
    week_df = df_between(df, corn_datetime - np.timedelta64(7,'D'), corn_datetime)
    day3_df = df_between(df, start_date=None, end_date=corn_datetime)
    day3_df = df_last(day3_df, 3)
    
    
    # Calculate features
    week_feats = calc_base_time_feats(week_df['<OPEN>'].values)
    day3_feats = calc_base_time_feats(day3_df['<OPEN>'].values)   
    m10_feats = calc_base_time_feats(m10_df['<OPEN>'].values)    
    m30_feats = calc_base_time_feats(m30_df['<OPEN>'].values)    
    
    week_support_feats = calc_support_feats(week_df['<OPEN>'].values)

    
    m30_rollings = calc_introday_rolling_aggs(m30_df)
    
    result = np.concatenate([
                            week_feats,
                            day3_feats,
                            m10_feats,
                            m30_feats,
                            week_support_feats,
                            m30_rollings
                            ], axis=0)
    

    # Form result
    result = np.expand_dims(result, axis=0)
    result = pd.DataFrame(result)
    
    result['ticker'] = ticker
    result['corn_datetime'] = corn_datetime
    
    return result

    

#####################################################



def load_single_feats(ticker2df, ticker_datetime):
    df = df_between(ticker2df[ticker_datetime['<TICKER>']], None, ticker_datetime['datetime'])
    df['corn_datetime'] = ticker_datetime['datetime']
    feats = calc_feats(df)

    return feats    


def load_feats(ticker2df, ticker_datetimes, n_jobs=10):
    p = Pool(n_jobs)
    feats_res = []
    res = p.starmap(load_single_feats, zip(repeat(ticker2df), ticker_datetimes))

    for feat in res:
        feats_res.append(feat)

    feats_res = pd.concat(feats_res, axis=0)
    feats_res = feats_res.infer_objects()
    feats_res.index=range(len(feats_res))
    
    return feats_res
      
    
def load_targets(ticker2df, ticker_datetimes, target_foo, n_jobs=10):
    p = Pool(n_jobs)
    res = p.starmap(target_foo, zip(repeat(ticker2df), ticker_datetimes))
    return np.array(res)
        
    
    

   