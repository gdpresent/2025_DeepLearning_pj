import json
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
import h5py
# !pip install parmap
import parmap
import os
import re
import time
from tqdm import tqdm
import warnings
from GD_utils.AI_tool.CNN_tool import process_params, process_imaging, OHLCV_cls
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    data_source = 'FnGuide'
    data_date = '20250320'
    COUNTRY = 'KR'
    image_version = f'v7' # v7: Suspension Transaction -> np.nan

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
        for from_day_type, to_day_type in [[20,20],[20,5],[5,5]]:
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