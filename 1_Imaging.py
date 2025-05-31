import json
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
import h5py
import parmap
import os
import re
import time
from tqdm import tqdm
import warnings
# from GD_utils.AI_tool.CNN_tool import process_params, process_imaging, OHLCV_cls
warnings.filterwarnings('ignore')

def read_pd_parquet(location):
    start = time.time()
    read = pd.read_parquet(location)
    print(f'Loading Complete({round((time.time() - start) / 60, 2)}min): {location}')
    return read
def save_as_pd_parquet(location, pandas_df_form):
    start = time.time()
    pandas_df_form.to_parquet(f'{location}')
    print(f'Saving Complete({round((time.time() - start) / 60, 2)}min): {location}')


def process_params(code, year_price, ExPost_return, from_day_type_int, to_day_type_int, image_size_dict, only_today=False):
    # code = 'A0015B0'
    # from_day_type_int, to_day_type_int = from_day_type, to_day_type
    local_params_list = []
    ExPost_ret_code = ExPost_return[code]
    price_a_stock = year_price[year_price.columns[year_price.columns.get_level_values(-1) == code]]#.dropna(how='all', axis=0)
    price_a_stock = price_a_stock.stack(1).droplevel(1).reindex(year_price.index)

    first_date = price_a_stock.dropna(how='all', axis=0).index[0]
    last_date = price_a_stock.dropna(how='all', axis=0).index[-1]
    price_a_stock = price_a_stock.loc[first_date:]

    if only_today:
        stt_idx = len(price_a_stock.loc[first_date:last_date].index)
    else:
        stt_idx = from_day_type_int*2 + 1

    for idx in range(stt_idx, len(price_a_stock.loc[first_date:last_date].index)+1):
        # input_df = price_a_stock.iloc[idx - from_day_type_int*2:idx]
        if (idx - from_day_type_int*2)<0:
            input_df = price_a_stock.copy()
        else:
            input_df = price_a_stock.iloc[idx - from_day_type_int*2:idx]
        if len(input_df.dropna(how='all', axis=0))==0:
            continue
        end_dt = input_df.index[-1]
        try:
            label = int(ExPost_ret_code.loc[end_dt] * 100_00_00) / 100_00_00
        except:
            label=-1
        input_df = input_df.reindex(['open', 'high', 'low', 'close', 'volume', 'C_MA', 'O_MA'], axis=1)
        chunk_size = from_day_type_int//to_day_type_int

        transformed_input = pd.DataFrame()
        for dt in input_df.index[chunk_size-1::chunk_size]:
            chunk_input = input_df.loc[:dt].iloc[-chunk_size:]

            o = chunk_input['open'].iloc[0]
            h = chunk_input['high'].max()
            l = chunk_input['low'].min()
            c = chunk_input['close'].iloc[-1]
            v = chunk_input['volume'].sum()
            transformed_input.loc[dt,['open','high','low','close','volume']] = [o,h,l,c,v]
        if len(transformed_input.dropna(how='all', axis=0))==0:
            continue
        open_time = pd.DataFrame(transformed_input.open.values, columns=['o_to_c'], index=[x+pd.DateOffset(hours=9) for x in transformed_input.open.index])
        close_time = pd.DataFrame(transformed_input.close.values, columns=['o_to_c'], index=[x+pd.DateOffset(hours=16) for x in transformed_input.close.index])
        open_to_close_time=pd.concat([open_time, close_time], axis=0).sort_index()
        C_MA_tmp=open_to_close_time.rolling(min_periods=to_day_type_int * 2, window=to_day_type_int * 2).mean().loc[lambda x: x.index.hour == 16]
        O_MA_tmp=open_to_close_time.rolling(min_periods=to_day_type_int * 2, window=to_day_type_int * 2).mean().loc[lambda x: x.index.hour == 9]

        C_MA_tmp.index = pd.to_datetime(C_MA_tmp.index.strftime("%Y-%m-%d"))
        O_MA_tmp.index = pd.to_datetime(O_MA_tmp.index.strftime("%Y-%m-%d"))

        transformed_input['C_MA'] =C_MA_tmp
        transformed_input['O_MA'] =O_MA_tmp

        #### 가격 전처리 들어가야하네....
        # 거래량 0 -> 거래량 NaN값 처리
        transformed_input.loc[transformed_input['volume']<=0, 'volume'] =np.nan

        # close값 없음 -> 모든 데이터 NaN값 처리(추론 불가능)
        transformed_input.loc[transformed_input['close'].isnull(), :] =np.nan

        # 시고저종 모두 NaN값 -> 거래량 NaN값 처리
        transformed_input.loc[
            (transformed_input['open'].isnull())&
            (transformed_input['high'].isnull())&
            (transformed_input['low'].isnull())&
            (transformed_input['close'].isnull()), 'volume'] = np.nan

        # 비어있는 high와 low는, 존재하는 open과 close로 filling
        # 이거는 여기에서 처리하면 안되겠다
        # --> 이미지 데이터 뽑아내기 직전에 처리했음(def process_code_idx)

        # low가 high보다 높으면 그날 모든 데이터 없다 처리(data error로 간주)
        transformed_input.loc[transformed_input['low']>transformed_input['high'], :] =np.nan

        # open값 없음 -> 오늘 종가/거래량, 어제 종가/거래량 존재하면 오늘의 open은 어제의 종가로 채워주자
        transformed_input.loc[
        (transformed_input['open'].isnull())&
        (transformed_input['close'].notna())&
        (transformed_input['volume'].notna())&
        (transformed_input['close'].notna().shift(1))&
        (transformed_input['volume'].notna().shift(1))
        , 'open'] = transformed_input['close'].shift(1)

        # 거래량 NaN -> 모든 가격 NaN값 처리
        transformed_input.loc[transformed_input['volume'].isnull(), :] =np.nan

        # 없다 처리 된 것들 있으니까 한 번 더
        # 시고저종 모두 NaN값 -> 거래량 NaN값 처리
        transformed_input.loc[
            (transformed_input['open'].isnull())&
            (transformed_input['high'].isnull())&
            (transformed_input['low'].isnull())&
            (transformed_input['close'].isnull()), 'volume'] = np.nan

        transformed_input['high'] = transformed_input[['open', 'high', 'close']].max(1)
        transformed_input['low'] = transformed_input[['open', 'low', 'close']].min(1)

        transformed_input = transformed_input.iloc[-to_day_type_int:]

        if (len(transformed_input.dropna(how='all', axis=0))<to_day_type_int)or(len(transformed_input.dropna(how='all', axis=1).columns)<len(transformed_input.columns)):
            continue
        local_params_list.append((transformed_input, code, end_dt, label, to_day_type_int, image_size_dict))
    return local_params_list
def convert_ohlcv_to_image(df, height, interval):
    # 원하는 데이터 크기의 행렬을 만들어 zero(black)를 다 깔아놓고 (0:black, 255:white)
    img = np.zeros((height, 3 * interval), dtype=np.uint8)

    # Price scaling
    min_price = np.nanmin([df['close'].min(), df['open'].min(), df['low'].min(), df[f'C_MA'].min(), df[f'O_MA'].min()])
    max_price = np.nanmax([df['close'].max(), df['open'].max(), df['high'].max(), df[f'C_MA'].max(), df[f'O_MA'].max()])
    max_volume = df['volume'].max()

    # 이미지 픽셀 값을 날짜 하나씩 채워 넣는 방식
    if max_price - min_price!=0:
        price_scale = (height * 4 / 5) / (max_price - min_price)
        volume_scale = (height / 5) / max_volume
        for i, (_, row) in enumerate(df.iterrows()):
            if not np.isnan(row['open']):
                open_pixel = int(np.round((row['open'] - min_price) * price_scale))
                img[-int((height * 1 / 5) + open_pixel), 3 * i] = 255
            if not np.isnan(row['close']):
                close_pixel = int(np.round((row['close'] - min_price) * price_scale))
                img[-int((height * 1 / 5) + close_pixel), 3*i + 2] = 255
            if not np.isnan(row['high']) and not np.isnan(row['low']):
                high_pixel = int(np.round((row['high'] - min_price) * price_scale))
                low_pixel = int(np.round((row['low'] - min_price)* price_scale))
                img[-int((height * 1 / 5) + high_pixel):-int((height * 1 / 5) + low_pixel)+1, 3 * i + 1] = 255
            if not np.isnan(row[f'O_MA']):
                Oma_pixel = int(np.round((row[f'O_MA'] - min_price) * price_scale))
                img[-int((height * 1 / 5) + Oma_pixel), 3*i] = 255
            if not np.isnan(row[f'C_MA']):
                Cma_pixel = int(np.round((row[f'C_MA'] - min_price) * price_scale))
                img[-int((height * 1 / 5) + Cma_pixel), 3*i+2] = 255
            if not np.isnan(row[f'C_MA']) and not np.isnan(row[f'O_MA']):
                Oma_pixel = int(np.round((row[f'O_MA'] - min_price) * price_scale))
                Cma_pixel = int(np.round((row[f'C_MA'] - min_price) * price_scale))
                ma_pixel = int((Oma_pixel+Cma_pixel)*0.5)
                img[-int((height * 1 / 5) + ma_pixel), 3*i+1] = 255
            if not np.isnan(row['volume']):
                volume_pixel = int(np.round(row['volume'] * volume_scale))
                if volume_pixel!=0:
                    img[-int(volume_pixel):, 3 * i + 1] = 255
    else:
        return None
    return img
def process_imaging(params):
    input_df, code, end_dt, label, day_type, image_size_dict = params
    img = convert_ohlcv_to_image(input_df, image_size_dict[day_type][0], image_size_dict[day_type][1])
    return img, code, end_dt.strftime("%Y%m%d"), label

class OHLCV_cls:
    def __init__(self, data_date, country, data_source):
        self.country = country
        self.data_source = data_source
        self.data_date = data_date
        self.excel_path = f'./data/{self.data_source}/excel'
        self.DB_path = f'./data/{self.data_source}/DB/{self.data_date}'
        os.makedirs(self.excel_path, exist_ok=True)
        os.makedirs(self.DB_path, exist_ok=True)

        print(f'{self.country} 데이터')
        print(f'데이터 기준 날짜: {self.data_date}')


        self.open, self.high, self.low, self.close, self.volume, self.code_to_name = self.get_daily_OHLCV()
        self.open_to_close = self.gen_Open_Close_concat()

        self.ExPost_dict = {
                            5: self.get_ExPost_return(n_day=5),
                            20: self.get_ExPost_return(n_day=20),
                            60: self.get_ExPost_return(n_day=60)
                            }
        self.close_MA_dict = {
                              5: self.get_Close_MA_n(n_day=5),
                              20:self.get_Close_MA_n(n_day=20),
                              60:self.get_Close_MA_n(n_day=60)
                             }
        self.open_MA_dict = {
                              5:self.get_Open_MA_n(n_day=5),
                              20:self.get_Open_MA_n(n_day=20),
                              60:self.get_Open_MA_n(n_day=60)
                             }

    # 가격 원본 데이터 Query
    def get_daily_OHLCV_KR_raw(self):
        if not os.path.exists(f'{self.DB_path}/{self.country}_mktcap_{self.data_date}.hd5'):
            tmp_read=pd.read_excel(f'{self.excel_path}/{self.country}_daily_data_{self.data_date}.xlsx', sheet_name='수정주가')
            _code_to_name = tmp_read.iloc[6:8].T.iloc[1:].set_index(6).rename_axis('code').rename(columns={7:'name'})
            _close = self.QuantiWise_data_preprocessing(tmp_read);print('close read')
            _open = self.QuantiWise_data_preprocessing(pd.read_excel(f'{self.excel_path}/{self.country}_daily_data_{self.data_date}.xlsx', sheet_name='수정시가'));print('open read')
            _high = self.QuantiWise_data_preprocessing(pd.read_excel(f'{self.excel_path}/{self.country}_daily_data_{self.data_date}.xlsx', sheet_name='수정고가'));print('high read')
            _low = self.QuantiWise_data_preprocessing(pd.read_excel(f'{self.excel_path}/{self.country}_daily_data_{self.data_date}.xlsx', sheet_name='수정저가'));print('low read')
            _volume = self.QuantiWise_data_preprocessing(pd.read_excel(f'{self.excel_path}/{self.country}_daily_data_{self.data_date}.xlsx', sheet_name='수정거래량'));print('volume read')
            _mktcap = self.QuantiWise_data_preprocessing(pd.read_excel(f'{self.excel_path}/{self.country}_daily_data_{self.data_date}.xlsx', sheet_name='시가총액'));print('volume read')
            _SuspTrnsctn = self.QuantiWise_data_preprocessing(pd.read_excel(f'{self.excel_path}/{self.country}_daily_data_{self.data_date}.xlsx', sheet_name='거래정지'));print('거래정지 read')

            save_as_pd_parquet(f'{self.DB_path}/{self.country}_code_to_name_{self.data_date}.hd5', _code_to_name)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_open_{self.data_date}.hd5', _open)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_high_{self.data_date}.hd5', _high)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_low_{self.data_date}.hd5', _low)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_volume_{self.data_date}.hd5', _volume)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_close_{self.data_date}.hd5', _close)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_mktcap_{self.data_date}.hd5', _mktcap)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_SuspTrnsctn_{self.data_date}.hd5', _SuspTrnsctn)
        else:
            _open = read_pd_parquet(f'{self.DB_path}/{self.country}_open_{self.data_date}.hd5')
            _high = read_pd_parquet(f'{self.DB_path}/{self.country}_high_{self.data_date}.hd5')
            _low = read_pd_parquet(f'{self.DB_path}/{self.country}_low_{self.data_date}.hd5')
            _volume = read_pd_parquet(f'{self.DB_path}/{self.country}_volume_{self.data_date}.hd5')
            _close = read_pd_parquet(f'{self.DB_path}/{self.country}_close_{self.data_date}.hd5')
            _mktcap = read_pd_parquet(f'{self.DB_path}/{self.country}_mktcap_{self.data_date}.hd5')
            _code_to_name = read_pd_parquet(f'{self.DB_path}/{self.country}_code_to_name_{self.data_date}.hd5')
            _SuspTrnsctn = read_pd_parquet(f'{self.DB_path}/{self.country}_SuspTrnsctn_{self.data_date}.hd5')
        _open = _open[_open>0]
        _high = _high[_high>0]
        _low = _low[_low>0]
        _volume = _volume[_volume>0]
        _close = _close[_close>0]

        close = _close.dropna(how='all', axis=0).dropna(how='all', axis=1)
        open = _open.dropna(how='all', axis=0).dropna(how='all', axis=1)
        high = _high.dropna(how='all', axis=0).dropna(how='all', axis=1)
        low = _low.dropna(how='all', axis=0).dropna(how='all', axis=1)
        volume = _volume.dropna(how='all', axis=0).dropna(how='all', axis=1)

        init_dt = np.nanmax([close.index.min(), open.index.min(), high.index.min(), low.index.min(), volume.index.min()])
        last_dt = np.nanmin([close.index.max(), open.index.max(), high.index.max(), low.index.max(), volume.index.max()])

        close = _close.loc[init_dt:last_dt]
        open = _open.loc[init_dt:last_dt]
        high = _high.loc[init_dt:last_dt]
        low = _low.loc[init_dt:last_dt]
        volume = _volume.loc[init_dt:last_dt]

        return open, high, low, close, volume, _code_to_name,_SuspTrnsctn

    # 가격 데이터 전처리
    def get_daily_OHLCV(self):
        open_raw, high_raw, low_raw, close_raw, volume_raw, code_to_name, SuspTrnsctn = self.get_daily_OHLCV_KR_raw()
        open_raw[SuspTrnsctn==1]=np.nan
        high_raw[SuspTrnsctn==1]=np.nan
        low_raw[SuspTrnsctn==1]=np.nan
        close_raw[SuspTrnsctn==1]=np.nan
        volume_raw[SuspTrnsctn==1]=np.nan
        # close_raw['A476470']
        if not os.path.exists(f'{self.DB_path}/{self.country}_close_processed_{self.data_date}.hd5'):
            # 거래량 0 -> 거래량 NaN값 처리
            volume_0_mask = volume_raw.le(0)
            # volume_raw.le(0).equals(volume_raw.eq(0))
            volume_raw[volume_0_mask] = np.nan

            # open값 없음 -> 오늘 종가/거래량, 어제 종가/거래량 존재하면 오늘의 open은 어제의 종가로 채워주자
            # 맨 뒤에서
            # open_null_mask = open_raw.isnull()
            # close_raw[open_null_mask]=np.nan
            # high_raw[open_null_mask]=np.nan
            # low_raw[open_null_mask]=np.nan
            # volume_raw[open_null_mask]=np.nan

            # close값 없음 -> 모든 데이터 NaN값 처리(추론 불가능)
            close_null_mask = close_raw.isnull()
            open_raw[close_null_mask] = np.nan
            high_raw[close_null_mask] = np.nan
            low_raw[close_null_mask] = np.nan
            volume_raw[close_null_mask] = np.nan

            # 시고저종 모두 NaN값 -> 거래량 NaN값 처리
            high_null_mask = high_raw.isnull()
            low_null_mask = low_raw.isnull()
            open_null_mask = open_raw.isnull()
            price_all_null_mask = open_null_mask & close_null_mask & high_null_mask & low_null_mask
            volume_raw[price_all_null_mask] = np.nan

            # 비어있는 high와 low는, 존재하는 open과 close로 filling
            # 이거는 여기에서 처리하면 안되겠다
            # --> 이미지 데이터 뽑아내기 직전에 처리했음(def process_code_idx)

            # low가 high보다 높으면 그날 모든 데이터 없다 처리(data error로 간주)
            low_exist_mask = low_raw.notna()
            high_exist_mask = high_raw.notna()
            low_exist = low_raw[low_exist_mask & high_exist_mask]
            high_exist = high_raw[low_exist_mask & high_exist_mask]
            Low_greater_than_high_mask = low_exist.gt(high_exist)
            open_raw[Low_greater_than_high_mask] = np.nan
            high_raw[Low_greater_than_high_mask] = np.nan
            low_raw[Low_greater_than_high_mask] = np.nan
            close_raw[Low_greater_than_high_mask] = np.nan
            volume_raw[Low_greater_than_high_mask] = np.nan

            # open값 없음 -> 오늘 종가/거래량, 어제 종가/거래량 존재하면 오늘의 open은 어제의 종가로 채워주자
            # open_null_mask 위에서 이미 한 번 선언 했었으니까
            today_close_exist = close_raw.notna()
            yesterday_close_exist = close_raw.notna().shift(1)
            today_open_null = open_raw.isnull()
            to_replace_value = close_raw.shift(1)[today_open_null & today_close_exist & yesterday_close_exist].stack()
            for idx in tqdm(to_replace_value.index, desc='######## NaN open filling ########'):
                open_raw.loc[idx[0], idx[1]] = to_replace_value.loc[idx]

            # 거래량 NaN -> 모든 가격 NaN값 처리
            volume_nan_mask = volume_raw.isnull()
            open_raw[volume_nan_mask] = np.nan
            high_raw[volume_nan_mask] = np.nan
            low_raw[volume_nan_mask] = np.nan
            close_raw[volume_nan_mask] = np.nan

            # 없다 처리 된 것들 있으니까 한 번 더
            # 시고저종 모두 NaN값 -> 거래량 NaN값 처리
            high_null_mask2 = high_raw.isnull()
            low_null_mask2 = low_raw.isnull()
            open_null_mask2 = open_raw.isnull()
            close_null_mask2 = close_raw.isnull()
            price_all_null_mask2 = open_null_mask2 & close_null_mask2 & high_null_mask2 & low_null_mask2
            volume_raw[price_all_null_mask2] = np.nan

            save_as_pd_parquet(f'{self.DB_path}/{self.country}_open_processed_{self.data_date}.hd5', open_raw)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_high_processed_{self.data_date}.hd5', high_raw)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_low_processed_{self.data_date}.hd5', low_raw)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_volume_processed_{self.data_date}.hd5', volume_raw)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_close_processed_{self.data_date}.hd5', close_raw)

        open_processed = read_pd_parquet(f'{self.DB_path}/{self.country}_open_processed_{self.data_date}.hd5')
        high_processed = read_pd_parquet(f'{self.DB_path}/{self.country}_high_processed_{self.data_date}.hd5')
        low_processed = read_pd_parquet(f'{self.DB_path}/{self.country}_low_processed_{self.data_date}.hd5')
        volume_processed = read_pd_parquet(f'{self.DB_path}/{self.country}_volume_processed_{self.data_date}.hd5')
        close_processed = read_pd_parquet(f'{self.DB_path}/{self.country}_close_processed_{self.data_date}.hd5')

        return open_processed, high_processed, low_processed, close_processed, volume_processed, code_to_name

    # 사후수익률 선계산
    def get_ExPost_return(self, n_day):
        if not os.path.exists(f'{self.DB_path}/{self.country}_ExPost_return_{n_day}_{self.data_date}.hd5'):
            tmp = self.open_to_close.pct_change((n_day*2)-1, fill_method=None).shift(-(n_day*2)).dropna(how='all', axis=0).loc[lambda x:x.index.hour==16]
            output=pd.DataFrame(tmp.values, columns=tmp.columns, index=[pd.to_datetime(x) for x in tmp.index.date])
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_ExPost_return_{n_day}_{self.data_date}.hd5', output)
        else:
            output = read_pd_parquet(f'{self.DB_path}/{self.country}_ExPost_return_{n_day}_{self.data_date}.hd5')
        return output

    # 시가-종가 연결
    def gen_Open_Close_concat(self):
        open_time = pd.DataFrame(self.open.values, columns=self.open.columns, index=[x+pd.DateOffset(hours=9) for x in self.open.index])
        close_time = pd.DataFrame(self.close.values, columns=self.close.columns, index=[x+pd.DateOffset(hours=16) for x in self.close.index])
        return pd.concat([open_time, close_time], axis=0).sort_index()

    # 시가-종가 연결하여 이동평균선 종가 부분을 만들어 놓음
    def get_Close_MA_n(self, n_day):
        if not os.path.exists(f'{self.DB_path}/{self.country}_C_MA_{n_day}_{self.data_date}.hd5'):
            tmp=self.open_to_close.rolling(min_periods=n_day*2, window=n_day*2).mean().loc[lambda x:x.index.hour==16]
            output=pd.DataFrame(tmp.values, columns=tmp.columns, index=[pd.to_datetime(x) for x in tmp.index.date])
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_C_MA_{n_day}_{self.data_date}.hd5', output)
        else:
            output = read_pd_parquet(f'{self.DB_path}/{self.country}_C_MA_{n_day}_{self.data_date}.hd5')
        return output
    # 시가-종가 연결하여 이동평균선 시가 부분을 만들어 놓음
    def get_Open_MA_n(self, n_day):
        if not os.path.exists(f'{self.DB_path}/{self.country}_O_MA_{n_day}_{self.data_date}.hd5'):
            tmp = self.open_to_close.rolling(min_periods=n_day * 2, window=n_day * 2).mean().loc[lambda x: x.index.hour == 9]
            output = pd.DataFrame(tmp.values, columns=tmp.columns, index=[pd.to_datetime(x) for x in tmp.index.date])
            # output_old = self.close.rolling(min_periods=n_day, window=n_day).mean()
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_O_MA_{n_day}_{self.data_date}.hd5', output)
        else:
            output = read_pd_parquet(f'{self.DB_path}/{self.country}_O_MA_{n_day}_{self.data_date}.hd5')
        return output

    # 1년단위로 데이터 불러오는 부분(주가데이터의 이미지화는 1년 단위로 처리함)
    def get_period_price_and_ExPostRet_v2(self, year_str, day_type):
        try:
            day_type_ago_date = self.volume.loc[f'{int(year_str) - 1}'].iloc[-int(day_type)*2:].index[0]
        except:
            day_type_ago_date = self.volume.loc[f'{int(year_str)}'].index[0]

        # 이제 모든 DataFrame을 병합합니다.
        year_price = pd.concat({
            'open': self.open.loc[day_type_ago_date:year_str].dropna(how='all', axis=1),
            'high': self.high.loc[day_type_ago_date:year_str].dropna(how='all', axis=1),
            'low': self.low.loc[day_type_ago_date:year_str].dropna(how='all', axis=1),
            'close': self.close.loc[day_type_ago_date:year_str].dropna(how='all', axis=1),
            'volume': self.volume.loc[day_type_ago_date:year_str].dropna(how='all', axis=1),
            'C_MA': self.close_MA_dict[day_type].loc[day_type_ago_date:year_str].dropna(how='all', axis=1),
            'O_MA': self.open_MA_dict[day_type].loc[day_type_ago_date:year_str].dropna(how='all', axis=1),
        }, axis=1)


        ExPost_return = self.ExPost_dict[day_type].loc[day_type_ago_date:year_str]
        return year_price, ExPost_return
    def get_period_price_and_ExPostRet_ETFonly(self, year_str, day_type, ETF_univ):
        try:
            day_type_ago_date = self.volume.loc[f'{int(year_str) - 1}'].iloc[-int(day_type)*2:].index[0]
        except:
            day_type_ago_date = self.volume.loc[f'{int(year_str)}'].index[0]

        # 이제 모든 DataFrame을 병합합니다.
        year_price = pd.concat({
            'open': self.open.loc[day_type_ago_date:year_str,ETF_univ].dropna(how='all', axis=1),
            'high': self.high.loc[day_type_ago_date:year_str,ETF_univ].dropna(how='all', axis=1),
            'low': self.low.loc[day_type_ago_date:year_str,ETF_univ].dropna(how='all', axis=1),
            'close': self.close.loc[day_type_ago_date:year_str,ETF_univ].dropna(how='all', axis=1),
            'volume': self.volume.loc[day_type_ago_date:year_str,ETF_univ].dropna(how='all', axis=1),
            'C_MA': self.close_MA_dict[day_type].loc[day_type_ago_date:year_str,ETF_univ].dropna(how='all', axis=1),
            'O_MA': self.open_MA_dict[day_type].loc[day_type_ago_date:year_str,ETF_univ].dropna(how='all', axis=1),
        }, axis=1)


        ExPost_return = self.ExPost_dict[day_type].loc[day_type_ago_date:year_str]
        return year_price, ExPost_return

    # Quantiwise 와꾸 데이터전처리
    def QuantiWise_data_preprocessing(self, data, univ=[]):
        data.columns = data.iloc[6]
        data = data.drop(range(0, 13), axis=0)
        data = data.rename(columns={'Code': 'date'}).rename_axis("종목코드", axis="columns").set_index('date')
        data.index = pd.to_datetime(data.index)
        if len(univ) != 0:
            data = data[univ]
        return data
    def Refinitiv_data_preprocessing(self, pvt_tmp):
        # pvt_tmp=o_pvt_raw.copy()
        pvt_tmp.index = pd.to_datetime(pvt_tmp.index)
        pvt_tmp = pvt_tmp.drop(pvt_tmp.iloc[0][pvt_tmp.iloc[0].apply(lambda x: type(x)==str)].index, axis=1)
        pvt_tmp = pvt_tmp[pvt_tmp.columns[~pvt_tmp.columns.isna()]]
        pvt_tmp = pvt_tmp.dropna(how='all', axis=0).dropna(how='all', axis=1)
        return pvt_tmp


if __name__ == '__main__':
    data_source = 'FnGuide'
    data_date = '20250527'
    COUNTRY = 'KR'
    image_version = f'v7'

    image_save_loc = f'./data/{data_source}/image/{COUNTRY}_{image_version}_All'
    os.makedirs(image_save_loc, exist_ok=True)

    self = OHLCV_cls(data_date=data_date, country=COUNTRY, data_source=data_source)

    image_size_dict={
                    5:[35,5],
                    10:[50,10],
                    20:[65,20],
                    60:[95,60],
                    }

    years=self.close.index.year.drop_duplicates().sort_values()
    # years = years[years>=2010]
    print(f'years: \n{years}')
    num_cores = mp.cpu_count()
    for year in years[::-1]:
        for from_day_type, to_day_type in [[20,20],[5,5]]:
            if os.path.exists(f"{image_save_loc}/{from_day_type}day_to_{to_day_type}day_{year}.h5"):
                print(f"################ PASS: {image_save_loc}/{from_day_type}day_to_{to_day_type}day_{year}.h5")
                continue

            year_str = str(year)

            start = time.time()
            year_price, ExPost_return = self.get_period_price_and_ExPostRet_v2(year_str, from_day_type)

            params_list = parmap.map(process_params, year_price['close'].columns,year_price,ExPost_return,from_day_type, to_day_type, image_size_dict,pm_parallel=True, pm_processes=num_cores, pm_pbar=True)
            params_list = [item for sublist in params_list for item in sublist]

            results = parmap.map(process_imaging, params_list, pm_parallel=True, pm_processes=num_cores, pm_pbar=True)

            with h5py.File(f"{image_save_loc}/{from_day_type}day_to_{to_day_type}day_{year}.h5", 'w') as hf:
                codes, dates, images_tmp = [], [], []
                for item in results:
                    if item[0] is not None:
                        images_tmp.append(item[0])
                        codes.append(item[1])
                        dates.append(item[2])
                        # labels_tmp.append(item[3])
                images = np.array(images_tmp)
                # labels = np.array(labels_tmp)

                # 이미지 데이터 저장 (lzf 압축 사용)
                hf.create_dataset("images", data=images, compression="lzf")

                # 문자열 리스트를 인코딩하여 바이너리 형태로 저장
                hf.create_dataset("codes", data=[s.encode('utf-8') for s in codes], compression="lzf")
                hf.create_dataset("dates", data=[s.encode('utf-8') for s in dates], compression="lzf")

                # labels는 실수 또는 정수 리스트로 저장되므로 dtype를 지정할 필요가 없습니다.
                # hf.create_dataset("labels", data=labels, compression="lzf")

            # 결과 출력
            print()
            print(f'{year} - {from_day_type}day to {to_day_type}day- images(ndim) :', images.shape)
            print(f'{year} - {from_day_type}day to {to_day_type}day- codes(ndim) :', len(codes))
            print(f'{year} - {from_day_type}day to {to_day_type}day- dates(ndim) :', len(dates))
            # print(f'{year} - {from_day_type}day to {to_day_type}day- labels(ndim) :', len(labels))
            print('#################################')