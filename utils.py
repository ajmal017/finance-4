from urllib.parse import urlencode
from urllib.request import urlopen
import time
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from parameters import ticker2finam





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
        df = pd.read_csv('{}/{}.csv'.format(dir_path, ticker))
        df['date'] = df.apply(lambda x: datetime.datetime.strptime(str(x['<DATE>']), "%Y%m%d"), axis=1)
        df['datetime'] = df.apply(lambda x: datetime.datetime.strptime(str(x['<DATE>'])+str(x['<TIME>'])[:4], "%Y%m%d%H%M"), axis=1)
        df['time'] = df['datetime'].apply(lambda x: x.time())

        if df.shape[0] < 300:
            continue

        ticker2df[ticker] = df


    return ticker2df


def sample_ticker_datetimes(ticker2df, min_date, max_date, cnt_by_ticker):
    ticker_datetimes = []
    for ticker in ticker2df:
        curr_ticker_datetimes = df_between(ticker2df[ticker], min_date, max_date)
        curr_ticker_datetimes = curr_ticker_datetimes[(curr_ticker_datetimes['time'] > datetime.time(hour=10, minute=20)) &\
                                                      (curr_ticker_datetimes['time'] < datetime.time(hour=18, minute=0))]

        if cnt_by_ticker is not None:
            curr_ticker_datetimes = curr_ticker_datetimes.sample(cnt_by_ticker)[['<TICKER>', 'datetime']].to_dict('records')
        else:
            curr_ticker_datetimes = curr_ticker_datetimes[['<TICKER>', 'datetime']].to_dict('records')
            
            
        ticker_datetimes.extend(curr_ticker_datetimes)

    return ticker_datetimes


def split_train_target(df, corn_datetime, target_interval):
    train_df = df_between(df, None, corn_datetime)
    train_df['corn_datetime'] = corn_datetime
    
    target_df = df_between(df, corn_datetime, corn_datetime + target_interval)
    target_df['corn_datetime'] = corn_datetime
    
    return train_df, target_df


def df_between(df, start_date=None, end_date=None):
    start_idx, end_idx = None, None
    if start_date is not None:
        start_idx = df['datetime'].searchsorted(start_date)
        
    if end_date is not None:
        end_idx = df['datetime'].searchsorted(end_date)

    return df[start_idx:end_idx]


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



























