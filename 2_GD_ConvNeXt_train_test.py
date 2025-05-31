import pandas as pd
import warnings
import torch
import os
from torch import nn, optim
import time
from tqdm import tqdm
import h5py
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.cuda.amp import autocast, GradScaler

# from GD_utils.AI_tool.CNN_tool import baseline_CNN_5day
# from GD_utils.AI_tool.CNN_tool import CustomDataset_all
# from GD_utils.AI_tool.CNN_tool import Train_Nepoch_ES, eval_loop
# from GD_utils.AI_tool.CNN_tool import read_pd_parquet
warnings.filterwarnings('ignore')
def save_as_pd_parquet(location, pandas_df_form):
    start = time.time()
    pandas_df_form.to_parquet(f'{location}')
    print(f'Saving Complete({round((time.time() - start) / 60, 2)}min): {location}')
def read_pd_parquet(location):
    start = time.time()
    read = pd.read_parquet(location)
    print(f'Loading Complete({round((time.time() - start) / 60, 2)}min): {location}')
    return read

class baseline_CNN_5day(nn.Module):
    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)  # underscore 추가
            m.bias.data.fill_(0.01)

    def __init__(self, dr_rate=0.5, stt_chnl=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=stt_chnl, out_channels=64, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 1)),
            )
        self.conv1.apply(self.init_weights)

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 3), padding=(2, 1)),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(),
                                   nn.MaxPool2d((2, 1)),
                                   )
        self.conv2.apply(self.init_weights)

        self.fc = nn.Linear(15360, 2)
        self.dropout = nn.Dropout(dr_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def loss_epoch_AMP(model, dataloader, criterion, DEVICE, optimizer=None, scaler=None):
    """
    AMP를 적용한 epoch 단위 loss 계산 함수
    (train 또는 val 구분은 optimizer 유무로 판단)
    """
    N = len(dataloader.dataset)
    running_loss = 0.0
    running_correct = 0

    # 모델이 train 모드인지, eval 모드인지 외부에서 설정해 준 후 호출
    for x_batch, y_batch in tqdm(dataloader):
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        # 자동 mixed precision
        with autocast():
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)

        if optimizer is not None:
            # 스케일러로 backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            # validation/test 모드일 땐 backward 안 함
            pass

        # accumulate loss
        running_loss += loss.item() * x_batch.shape[0]

        # accuracy
        preds = y_hat.argmax(dim=1)
        running_correct += torch.sum(preds == y_batch).item()

    loss_epoch = running_loss / N
    accuracy_epoch = running_correct / N * 100
    return loss_epoch, accuracy_epoch, running_correct
def Train_Nepoch_ES_AMP(model, train_DL, val_DL, criterion, DEVICE, optimizer, EPOCH, BATCH_SIZE, TRAIN_RATIO, save_model_path, save_history_path, N_EPOCH_ES, init_val_loss=1e20, **kwargs):
    """
    AMP + Early Stopping 적용된 학습 루프
    """
    # AMP를 위한 GradScaler
    scaler = GradScaler()

    # 스케줄러 필요시
    if "LR_STEP" in kwargs:
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=kwargs["LR_STEP"], gamma=kwargs["LR_GAMMA"])
    else:
        scheduler = None

    loss_history = {"train": [], "val": []}
    acc_history = {"train": [], "val": []}

    print(f'Initial Validation Loss: {init_val_loss}')
    no_improve_count = 0

    # 메모리 포맷 변경 (opt) - 모델을 channels_last로 변경
    # model = model.to(memory_format=torch.channels_last)

    for ep in range(EPOCH):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {ep+1}/{EPOCH}] LR={current_lr}")

        # -------------------
        #  (1) Train Mode
        # -------------------
        model.train()
        train_loss, train_acc, _ = loss_epoch_AMP(
            model, train_DL, criterion, DEVICE, optimizer, scaler
        )
        loss_history["train"].append(train_loss)
        acc_history["train"].append(train_acc)

        # -------------------
        #  (2) Validation Mode
        # -------------------
        model.eval()
        with torch.no_grad():
            val_loss, val_acc, _ = loss_epoch_AMP(
                model, val_DL, criterion, DEVICE, optimizer=None, scaler=None
            )
            loss_history["val"].append(val_loss)
            acc_history["val"].append(val_acc)

        # 스케줄러가 있으면
        if scheduler is not None:
            scheduler.step()

        # Early Stopping 체크
        if val_loss < init_val_loss:
            init_val_loss = val_loss
            print(f"--> best val loss updated: {round(init_val_loss, 5)}")
            no_improve_count = 0
            # 모델 가중치 저장
            torch.save(model.state_dict(), save_model_path)
        else:
            no_improve_count += 1

        print(f"[Train] loss={train_loss:.5f}, acc={train_acc:.2f} | "
              f"[Val] loss={val_loss:.5f}, acc={val_acc:.2f}  "
              f"(no_improve_count={no_improve_count}) | time: {round(time.time() - epoch_start)}s")
        print("-"*50)

        if no_improve_count >= N_EPOCH_ES:
            print("Early stopping triggered.")
            break

    # 학습 이력 저장
    torch.save({
        "loss_history": loss_history,
        "acc_history": acc_history,
        "EPOCH": EPOCH,
        "BATCH_SIZE": BATCH_SIZE,
        "TRAIN_RATIO": TRAIN_RATIO
    }, save_history_path)

    return acc_history['train'][-1], acc_history['val'][-1], len(acc_history['train'])
def eval_loop(dataloader, net, loss_fn, DEVICE):
    # dataloader, net, loss_fn, DEVICE=test_DL, model_V2_TmPlng_body2, criterion, DEVICE
    running_loss = 0.0
    current = 0
    net.eval()
    predict = []
    codes=[]
    dates=[]
    returns=[]
    target = []
    with torch.no_grad():
        with tqdm(dataloader) as t:
            for batch, (img, code, date, rets, label) in enumerate(t):
                X = img.to(DEVICE)
                y = label.to(DEVICE)
                y_pred = net(X)
                target.append(y.detach())
                codes.extend(code)
                dates.extend(date)
                returns.append(rets.detach())
                predict.append(y_pred.detach())
                loss = loss_fn(y_pred, y.long())
                running_loss += loss.item() * len(X)  # Here we update the running_loss
                avg_loss = running_loss / (current + len(X))
                t.set_postfix({'running_loss': avg_loss})
                current += len(X)
    returns = torch.cat(returns).cpu().numpy()
    targets = torch.cat(target).cpu().numpy()
    return avg_loss, torch.cat(predict),codes, dates,returns,targets

def labeling_v1(lb_tmp):
    if lb_tmp>0:
        label = 1
    else:
        label = 0
    return label
class CustomDataset_all(Dataset):
    def __init__(self, image_data_path, data_source, train, data_date,stt_date=None, until_date=None, transform=None,Pred_Hrz = 20,cap_criterion=0.0,F_day_type=20, T_day_type=20):
        self.transform = transform
        self.train=train

        if "KR" in image_data_path: self.country ="KR"
        else:self.country ="US"

        self.data_source = data_source
        self.data_date = data_date
        self.excel_path = f'./data/{self.data_source}/excel'
        self.DB_path = f'./data/{self.data_source}/DB/{self.data_date}'

        until_date = pd.to_datetime(until_date)
        until_year = until_date.year
        stt_date = pd.to_datetime(stt_date)

        if stt_date == None:
            stt_year = 1990
        else:
            stt_year = stt_date.year
        years = [x for x in range(until_year+1) if x>=stt_year]

        ExPost = self.get_ExPost_return(n_day=Pred_Hrz)
        self.last_date = ExPost.index[-1]
        date_code_list_MiniCap_remove = self.get_date_code_list_MiniCap_remove(until_date, stt_date,cap_criterion=cap_criterion)

        self.data = []
        self.codes = []
        self.dates = []
        self.returns = []
        self.labels = []
        for year in tqdm(years[::-1], desc='### Data Loading ###'):
            if not os.path.exists(f"{image_data_path}/{F_day_type}day_to_{T_day_type}day_{year}.h5"):
                print(f'파일 없음:\n\t"{image_data_path}/{F_day_type}day_to_{T_day_type}day_{year}.h5"')
                continue
            with h5py.File(f"{image_data_path}/{F_day_type}day_to_{T_day_type}day_{year}.h5", 'r') as hf:
                loaded_images = hf['images'][:]
                loaded_codes = [s.decode('utf-8') for s in hf['codes'][:]]
                loaded_dates = [s.decode('utf-8') for s in hf['dates'][:]]
                for img, code, date in zip(loaded_images, loaded_codes, loaded_dates):
                    if f'{date}_{code}' in date_code_list_MiniCap_remove:
                        ret = ExPost.loc[pd.to_datetime(date), code]
                        lab = labeling_v1(ret)

                        self.data.append(img)
                        self.codes.append(code)
                        self.dates.append(date)
                        self.returns.append(ret)
                        self.labels.append(lab)
        if self.train:
            # clear memory
            self.codes = []
            self.dates = []
            self.returns = []
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        if self.train:
            return img, label
        else:
            code = self.codes[idx]
            date = self.dates[idx]
            returns = self.returns[idx]
            return img, code, date, returns, label
    def get_date_code_list_MiniCap_remove(self, until_date,stt_date,cap_criterion):
        # if os.path.exists(f'./data/{data_source}/DB/{data_date}/{COUNTRY}_mktcap.hd5'):
        if os.path.exists(f'{self.DB_path}/{self.country}_mktcap_{self.data_date}.hd5'):
            # Cap_df_raw = read_pd_parquet(f'./data/{data_source}/DB/{data_date}/{COUNTRY}_mktcap.hd5')#.loc[stt_date:f'{until_date}']
            Cap_df_raw = read_pd_parquet(f'{self.DB_path}/{self.country}_mktcap_{self.data_date}.hd5')#.loc[stt_date:f'{until_date}']
        else:
            # Cap_df_raw = data_preprocessing(pd.read_excel(f'./data/{data_source}/{COUNTRY}_daily_data_{data_date}.xlsx', sheet_name='시가총액'))
            # save_as_pd_parquet(f'./data/{data_source}/{COUNTRY}_MktCap_{data_date}.hd5', Cap_df_raw)
            raise ValueError('데이터 없음')
        Cap_df_raw = Cap_df_raw.loc[:f'{self.last_date}']
        Cap_df_raw = Cap_df_raw.loc[stt_date:f'{until_date}']

        Cap_df_removed_20th = Cap_df_raw[Cap_df_raw.rank(ascending=True, pct=True, axis=1) >= cap_criterion]
        # print(f'Cap_df:\n{Cap_df_removed_20th}')

        dt_code_set = set()
        for dt, code in tqdm(Cap_df_removed_20th.stack().index, desc=f'시가총액 하위 {int(cap_criterion * 100)}% 종목 제거 리스트 산출'):
            dt_code_set.add(f'{dt.strftime("%Y%m%d")}_{code}')
        return dt_code_set
    def get_ExPost_return(self,n_day):
        if not os.path.exists(f'{self.DB_path}/{self.country}_ExPost_return_{n_day}_{self.data_date}.hd5'):
            tmp = self.open_to_close.pct_change((n_day*2)-1, fill_method=None).shift(-(n_day*2)).dropna(how='all', axis=0).loc[lambda x:x.index.hour==16]
            output=pd.DataFrame(tmp.values, columns=tmp.columns, index=[pd.to_datetime(x) for x in tmp.index.date])
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_ExPost_return_{n_day}_{self.data_date}.hd5', output)
        else:
            output = read_pd_parquet(f'{self.DB_path}/{self.country}_ExPost_return_{n_day}_{self.data_date}.hd5')
        return output


if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)
    data_source = 'FnGuide'

    COUNTRY = 'KR'
    data_date = f'20250320'

    image_path = f'./data/{data_source}/image/{COUNTRY}_All'
    DB_path = f'./data/{data_source}/DB/{data_date}'

    transform = transforms.ToTensor()

    # 각종 Hyper-parameters 설정
    criterion = nn.CrossEntropyLoss()
    '2_GD_ConvNeXt_train_test'
    model_name = 'ConvNeXt'
    model_save_path = f"./models/{COUNTRY}_{model_name}"
    os.makedirs(model_save_path, exist_ok=True)

    Max_EPOCH = 1000
    cap_cut=0.0
    BATCH_SIZE = 128
    MaxTry = 3
    LR_pow = 5
    DR = 50
    TRAIN_RATIO= 0.7

    CAP = int(cap_cut*100)
    LR = 1 / (10 ** LR_pow)
    dr_rate = DR/100

    SAVE_CONCAT = pd.DataFrame()
    Cap_df_raw = read_pd_parquet(f'{DB_path}/KR_mktcap_{data_date}.hd5').sort_index()
    business_dates = Cap_df_raw[[Cap_df_raw.notna().sum().sort_values().index[-1]]]
    monthly_dates = business_dates.groupby(pd.Grouper(freq='BM')).tail(1)#.iloc[:-1]
    monthly_dates = monthly_dates.loc["2010-06":data_date]
    monthly_dates = monthly_dates.loc[monthly_dates.index[monthly_dates.index.month.isin([6,12])]]
    monthly_dates = monthly_dates.reset_index().assign(next_month=lambda x: x['date'].shift(-1))[['date', 'next_month']]  # .applymap(lambda x:pd.to_datetime(x))

    learn_DATE = '2022-12-31'
    test_DATE = '2025-04-30'

    learn_DATE_str = pd.to_datetime(learn_DATE).strftime("%Y%m%d")
    test_DATE_str = pd.to_datetime(test_DATE).strftime("%Y%m%d")

    print('learn DATE: ', learn_DATE_str)
    print('next DATE: ', test_DATE_str)


    dataset_050505 = CustomDataset_all(image_data_path=image_path,
                                       train=True,
                                       data_date=data_date,
                                       data_source=data_source,
                                       F_day_type=5,
                                       T_day_type=5,
                                       Pred_Hrz=5,
                                       until_date=learn_DATE,
                                       stt_date=learn_DATE - pd.DateOffset(years=30),
                                       cap_criterion=cap_cut,
                                       transform=transform
                                       )
    for i in range(1, 6):
        Randomly_Stratified_050505 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - TRAIN_RATIO))
        train_idx_050505, val_idx_050505 = next(Randomly_Stratified_050505.split(list(range(len(dataset_050505))), dataset_050505.labels))
        train_DS_050505 = Subset(dataset_050505, train_idx_050505)
        val_DS_050505 = Subset(dataset_050505, val_idx_050505)

        train_DL_050505 = DataLoader(train_DS_050505, batch_size=BATCH_SIZE, shuffle=True);
        val_DL_050505 = DataLoader(val_DS_050505, batch_size=BATCH_SIZE, shuffle=True)

        print(f'learn_date: {learn_DATE_str}')
        print(f'cap_cut: {CAP}')
        print(f'BATCH SIZE: {BATCH_SIZE}')
        print(f'MaxTry: {MaxTry}')
        print(f'LR_pow: {LR_pow}')
        print(f'iteration: {i}')

        bCNN_050505_mdl_pth = f"{model_save_path}/{model_name}_050505_{learn_DATE_str}_{i}.pt"
        bCNN_050505_hry_pth = f"{model_save_path}/{model_name}_050505_{learn_DATE_str}_{i}_hist.pt"

        bCNN_050505_model = nn.DataParallel(baseline_CNN_5day(dr_rate=dr_rate, stt_chnl=1)).to(DEVICE)
        bCNN_050505_model_latest_val_loss = 100E100

        # optimizer 설정해서
        bCNN_050505_optr = optim.Adam(bCNN_050505_model.parameters(), lr=LR)

        print('================bCNN_050505================\n' * 1)
        bCNN_050505_Tacc, bCNN_050505_Vacc, bCNN_050505_eps = Train_Nepoch_ES_AMP(bCNN_050505_model,
                                                                                  train_DL_050505,
                                                                                  val_DL_050505,
                                                                                  criterion,
                                                                                  DEVICE,
                                                                                  bCNN_050505_optr,
                                                                                  Max_EPOCH, BATCH_SIZE,
                                                                                  TRAIN_RATIO,
                                                                                  bCNN_050505_mdl_pth,
                                                                                  bCNN_050505_hry_pth,
                                                                                  MaxTry,
                                                                                  bCNN_050505_model_latest_val_loss
                                                                                  )

        bCNN_050505_model = nn.DataParallel(baseline_CNN_5day(dr_rate=dr_rate, stt_chnl=1)).to(DEVICE)
        bCNN_050505_model.load_state_dict(torch.load(bCNN_050505_mdl_pth, map_location=DEVICE))

        test_DL_050505 = DataLoader(CustomDataset_all(
                                                      image_path,
                                                      train=False,
                                                      data_date=data_date,
                                                      data_source=data_source,
                                                      F_day_type=5, T_day_type=5,
                                                      stt_date=learn_DATE + pd.DateOffset(days=1),
                                                      until_date=test_DATE,
                                                      Pred_Hrz=5,
                                                      cap_criterion=cap_cut,
                                                      transform=transform
                                                      ), batch_size=256, shuffle=False)

        bCNN_050505_avg_loss, \
        bCNN_050505_preds_tmp,\
        bCNN_050505_codes,\
        bCNN_050505_dates,\
        bCNN_050505_returns,\
        bCNN_050505_labels = eval_loop(test_DL_050505, bCNN_050505_model, criterion, DEVICE)

        # 예측 결과를 최대 확률의 인덱스로 변환
        bCNN_050505_preds = torch.argmax(torch.nn.Softmax(dim=1)(bCNN_050505_preds_tmp), dim=1).cpu().numpy()
        bCNN_050505_1preds = torch.nn.Softmax(dim=1)(bCNN_050505_preds_tmp)[:, 1].cpu().numpy()

        # Accuracy= 올바른 예측 수 / 전체 예측 수 -> 내가 베팅한 것 중 몇 개 적중했냐
        bCNN_050505_acc = accuracy_score(bCNN_050505_labels, bCNN_050505_preds)

        # Prediction = 참 양성(TP) / (참 양성(TP) + 거짓 양성(FP)) -> 내가 오를거라 베팅한것 중 진짜 오른것은 몇 개냐
        # 정밀도가 높을수록 모델의 성능이 좋다
        bCNN_050505_prec = precision_score(bCNN_050505_labels, bCNN_050505_preds)

        # Recall = 참 양성(TP) / (참 양성(TP) + 거짓 음성(FN)) ->실제로 양성인 샘플 중에서 모델이 양성으로 정확히 예측한 샘플의 비율
        # 재현율이 높을수록 모델의 성능이 좋다
        bCNN_050505_rcll = recall_score(bCNN_050505_labels, bCNN_050505_preds)

        # F1 = Prediction와 Recall의 조화평균
        bCNN_050505_f1 = f1_score(bCNN_050505_labels, bCNN_050505_preds)

        print(f'{"=" * 20} bCNN_050505{"=" * 20}')
        print(f'bCNN_050505 labels: {int(100 * (sum(bCNN_050505_labels) / len(bCNN_050505_labels)))}%')
        print(f'bCNN_050505 predicts: {int(100 * (sum(bCNN_050505_preds) / len(bCNN_050505_preds)))}%')
        print(f'bCNN_050505 Accuracy: {int(10000 * (bCNN_050505_acc)) / 100}%')
        print(f'bCNN_050505 Precision: {int(10000 * (bCNN_050505_prec)) / 100}%')
        print(f'bCNN_050505 Recall: {int(10000 * (bCNN_050505_rcll)) / 100}%')
        print(f'bCNN_050505 F1-score: {int(10000 * (bCNN_050505_f1)) / 100}%')