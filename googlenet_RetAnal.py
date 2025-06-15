import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torch import nn, optim
import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report,)

from datetime import datetime
from tqdm import tqdm

import os
import pandas as pd
import numpy as np
import h5py
import time
import warnings
warnings.filterwarnings('ignore')

# os.makedirs("C:/Users/flydc/jupyter_project/pjt1_deep_learning/models/KR_GoogLeNet", exist_ok=True)

# =========================
# [UTIL] Parquet 파일 저장/로딩
# =========================
def save_as_pd_parquet(location, pandas_df_form):
    start = time.time()
    pandas_df_form.to_parquet(f'{location}')
    print(f'Saving Complete({round((time.time() - start) / 60, 2)}min): {location}')

def read_pd_parquet(location):
    start = time.time()
    read = pd.read_parquet(location)
    print(f'[LOAD OK] {location}')
    return read

"""================
첫번째 모델 : SeNet
================"""

# =========================
# [SE Block] 채널별 특성 강조
# =========================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# =========================
# [CNN 모델] SEBlock 포함 VGG-style 구조
# =========================
class SENet_CNN_5day(nn.Module):
    def __init__(self, dr_rate=0.5, stt_chnl=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(stt_chnl, 64, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            SEBlock(64),
            nn.MaxPool2d((2, 1))
        )
        self.conv1.apply(self.init_weights)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            SEBlock(128),
            nn.MaxPool2d((2, 1))
        )
        self.conv2.apply(self.init_weights)

        self.dropout = nn.Dropout(dr_rate)
        self.fc = nn.Linear(15360, 2)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


"""================
두번째 모델 : GoogleNet
================"""
# =========================
# [GoogLeNet 구성 블록] Inception 모듈 정의
# =========================
class Inception(nn.Module):
    def __init__(self, in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, out1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red3x3, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(red3x3, out3x3, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red5x5, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(red5x5, out5x5, kernel_size=5, padding=2),
            nn.ReLU(True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], 1)

# =========================
# [CNN 모델] GoogLeNet 구조 정의 (aux_logits=False)
# =========================

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], 1)
class GoogLeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, dropout=0.4):
        super().__init__()
        self.conv1 = BasicConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# =========================
# [Wrapper] GoogLeNet_CNN_5day (기존 구조 호환)
# =========================
class GoogLeNet_CNN_5day(nn.Module):
    def __init__(self, dr_rate=0.5, stt_chnl=1):
        super().__init__()
        self.model = GoogLeNet(in_channels=stt_chnl, num_classes=2, dropout=dr_rate)

    def forward(self, x):
        return self.model(x)



# =========================
# [라벨링 함수] 수익률 기준 이진 라벨링
# =========================
def labeling_v1(lb_tmp):
    return 1 if lb_tmp > 0 else 0

# =========================
# [AMP Epoch Loss] (Train/Val)
# =========================
def loss_epoch_AMP(model, dataloader, criterion, DEVICE, optimizer=None, scaler=None):
    N = len(dataloader.dataset)
    running_loss = 0.0
    running_correct = 0

    for x_batch, y_batch in tqdm(dataloader):
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        with autocast():
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)

        if optimizer is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * x_batch.shape[0]
        preds = y_hat.argmax(dim=1)
        running_correct += torch.sum(preds == y_batch).item()

    loss_epoch = running_loss / N
    accuracy_epoch = running_correct / N * 100
    return loss_epoch, accuracy_epoch, running_correct

# =========================
# [Training Loop with Early Stopping + AMP]
# =========================
def Train_Nepoch_ES_AMP(model, train_DL, val_DL, criterion, DEVICE, optimizer, EPOCH, BATCH_SIZE,
                        TRAIN_RATIO, save_model_path, save_history_path, N_EPOCH_ES, init_val_loss=1e20, **kwargs):
    scaler = GradScaler()

    if "LR_STEP" in kwargs:
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=kwargs["LR_STEP"], gamma=kwargs["LR_GAMMA"])
    else:
        scheduler = None

    loss_history = {"train": [], "val": []}
    acc_history = {"train": [], "val": []}
    print(f'Initial Validation Loss: {init_val_loss}')
    no_improve_count = 0

    for ep in range(EPOCH):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {ep+1}/{EPOCH}] LR={current_lr}")

        model.train()
        train_loss, train_acc, _ = loss_epoch_AMP(model, train_DL, criterion, DEVICE, optimizer, scaler)
        loss_history["train"].append(train_loss)
        acc_history["train"].append(train_acc)

        model.eval()
        with torch.no_grad():
            val_loss, val_acc, _ = loss_epoch_AMP(model, val_DL, criterion, DEVICE)

        loss_history["val"].append(val_loss)
        acc_history["val"].append(val_acc)

        if scheduler is not None:
            scheduler.step()

        if val_loss < init_val_loss:
            init_val_loss = val_loss
            print(f"--> best val loss updated: {round(init_val_loss, 5)}")
            no_improve_count = 0
            torch.save(model.state_dict(), save_model_path)
        else:
            no_improve_count += 1

        print(f"[Train] loss={train_loss:.5f}, acc={train_acc:.2f} | "
              f"[Val] loss={val_loss:.5f}, acc={val_acc:.2f}  "
              f"(no_improve_count={no_improve_count}) | time: {round(time.time() - epoch_start)}s")
        print("-" * 50)

        if no_improve_count >= N_EPOCH_ES:
            print("Early stopping triggered.")
            break

    torch.save({
        "loss_history": loss_history,
        "acc_history": acc_history,
        "EPOCH": EPOCH,
        "BATCH_SIZE": BATCH_SIZE,
        "TRAIN_RATIO": TRAIN_RATIO
    }, save_history_path)

    return acc_history['train'][-1], acc_history['val'][-1], len(acc_history['train'])

# =========================
# [Test Loop]
# =========================
def eval_loop(dataloader, net, loss_fn, DEVICE):
    net.eval()
    running_loss = 0.0
    current = 0
    predict = []
    codes, dates, returns, target = [], [], [], []

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
                running_loss += loss.item() * len(X)
                avg_loss = running_loss / (current + len(X))
                t.set_postfix({'running_loss': avg_loss})
                current += len(X)

    returns = torch.cat(returns).cpu().numpy()
    targets = torch.cat(target).cpu().numpy()
    return avg_loss, torch.cat(predict), codes, dates, returns, targets

# =========================
# [데이터셋 클래스] 이미지 + 메타데이터 조합
# =========================
class CustomDataset_all(Dataset):
    def __init__(self, image_data_path, DB_path, data_source, train, data_date,
                 stt_date=None, until_date=None, transform=None,
                 Pred_Hrz=20, cap_criterion=0.0, F_day_type=20, T_day_type=20, country="KR"):
        self.transform = transform
        self.train = train
        self.image_data_path = image_data_path
        self.DB_path = DB_path
        self.data_source = data_source
        self.data_date = data_date
        self.country = country

        until_date = pd.to_datetime(until_date)
        stt_date = pd.to_datetime(stt_date)
        years = [x for x in range(stt_date.year, until_date.year + 1)]

        self.ExPost = self.get_ExPost_return(Pred_Hrz)
        self.last_date = self.ExPost.index[-1]
        self.date_code_list_MiniCap_remove = self.get_date_code_list_MiniCap_remove(until_date, stt_date, cap_criterion)

        self.data, self.codes, self.dates, self.returns, self.labels = [], [], [], [], []
        for year in tqdm(years[::-1], desc='### Data Loading ###'):
            file_path = f"{image_data_path}/{F_day_type}day_to_{T_day_type}day_{year}.h5"
            if not os.path.exists(file_path):
                print(f'파일 없음: {file_path}')
                continue
            with h5py.File(file_path, 'r') as hf:
                images = hf['images'][:]
                codes = [s.decode('utf-8') for s in hf['codes'][:]]
                dates = [s.decode('utf-8') for s in hf['dates'][:]]
                for img, code, date in zip(images, codes, dates):
                    if f'{date}_{code}' in self.date_code_list_MiniCap_remove:
                        ret = self.ExPost.loc[pd.to_datetime(date), code]
                        lab = labeling_v1(ret)
                        self.data.append(img)
                        self.codes.append(code)
                        self.dates.append(date)
                        self.returns.append(ret)
                        self.labels.append(lab)

        if self.train:
            self.codes, self.dates, self.returns = [], [], []  # 학습 시 메모리 최소화

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
            return img, self.codes[idx], self.dates[idx], self.returns[idx], label

    def get_ExPost_return(self, n_day):
        path = f'{self.DB_path}/{self.country}_ExPost_return_{n_day}_{self.data_date}.hd5'
        if not os.path.exists(path):
            raise ValueError(f"ExPost 파일 없음: {path}")
        return read_pd_parquet(path)

    def get_date_code_list_MiniCap_remove(self, until_date, stt_date, cap_criterion):
        path = f'{self.DB_path}/{self.country}_mktcap_{self.data_date}.hd5'
        if not os.path.exists(path):
            raise ValueError(f"시가총액 파일 없음: {path}")
        Cap_df_raw = read_pd_parquet(path)
        Cap_df_raw = Cap_df_raw.loc[stt_date:until_date]
        Cap_df_filtered = Cap_df_raw[Cap_df_raw.rank(pct=True, axis=1) >= cap_criterion]

        dt_code_set = set()
        for dt, code in tqdm(Cap_df_filtered.stack().index, desc=f'시가총액 하위 {int(cap_criterion * 100)}% 종목 제거 리스트'):
            dt_code_set.add(f'{dt.strftime("%Y%m%d")}_{code}')
        return dt_code_set
def inference_result_save(pred1, test_code, test_date, test_return, test_label, epoches):
    return pd.DataFrame(
        {
            "Prob_Positive": pred1,
            "종목코드": test_code,
            "return": test_return,
            "label": test_label,
            "epoch": epoches
        }, index=pd.to_datetime(test_date)).rename_axis("date").sort_index()
from typing import Dict
def count_parameters(
    model: torch.nn.Module,
    by_layer: bool = False,
    require_grad_only: bool = False,
) -> Dict[str, int | float | pd.DataFrame]:
    """
    모델 파라미터 개수 및 메모리 사용량(FP32) 계산.

    Parameters
    ----------
    model : nn.Module
        대상 모델. `DataParallel` 인 경우 model.module 로 자동 전개.
    by_layer : bool, default False
        True → layer별 개수를 `pandas.DataFrame` 으로 추가 반환.
    require_grad_only : bool, default False
        True → `requires_grad=True` 인 텐서만 집계 (freeze‑fine‑tuning 시 유용).

    Returns
    -------
    Dict[str, Any]
        ├─ total_params       (int) : 전체 파라미터 수
        ├─ trainable_params   (int) : requires_grad=True 파라미터 수
        ├─ total_memory(MB)   (float) : FP32 기준 메모리 사용량
        └─ layer_details      (DataFrame) : by_layer=True 일 때만 포함
    """
    # DataParallel → 실제 모델 꺼내기
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    param_iter = (
        (n, p) for n, p in model.named_parameters()
        if (p.requires_grad or not require_grad_only)
    )

    total = 0
    trainable = 0
    layer_rows = []

    for name, p in param_iter:
        numel = p.numel()
        total += numel
        if p.requires_grad:
            trainable += numel
        if by_layer:
            layer_rows.append({
                "layer": name,
                "num_params": numel,
                "trainable": p.requires_grad
            })

    memory_mb = total * 4 / 1024 ** 2  # float32: 4 Byte

    result: Dict[str, int | float | pd.DataFrame] = {
        "total_params": total,
        "trainable_params": trainable,
        "total_memory(MB)": round(memory_mb, 2),
    }
    if by_layer:
        result["layer_details"] = pd.DataFrame(layer_rows)

    return result
if __name__ == "__main__":
    # [Step1] 기본 설정
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {DEVICE}")

    # [Step2] 하이퍼파라미터
    TRAIN_RATIO = 0.7
    BATCH_SIZE = 128

    LR_pow = 4
    LR = 1 / (10 ** LR_pow)  # = 1e-4

    EPOCHS = 1000            # Max_EPOCH
    N_EPOCH_ES = 3           # MaxTry

    DR = 50
    dr_rate = DR / 100       # 모델 생성 시 적용됨

    """
    # [Step3] 날짜 설정
    learn_DATE = pd.to_datetime('2022-12-31')
    test_DATE = pd.to_datetime('2023-12-31')
    stt_DATE = learn_DATE - pd.DateOffset(years=1)  # 최근 1년만 사용
    """
     #전체 데이터 학습 시 사용
    learn_DATE = pd.to_datetime('2023-12-31')
    test_DATE = pd.to_datetime('2025-04-30')
    stt_DATE = pd.to_datetime('2022-01-01')



    # [Step4] 경로 설정
    data_source = 'FnGuide'
    COUNTRY = 'KR'
    data_date = '20250527'
    image_path = f'./data/{data_source}/image/{COUNTRY}_v1_All'
    DB_path = f'./data/{data_source}/DB/{data_date}'
    data_date = "20250527"
    cap_criterion = 0.0
    CodeName=pd.read_excel(f'./data/FnGuide/excel/KR_daily_data_{data_date}.xlsx', sheet_name='보통주', index_col=0)


    # [Step5] 이미지 변환
    transform = transforms.ToTensor()

    # [Step6] 학습용 데이터셋 로딩
    dataset = CustomDataset_all(
        image_data_path=image_path,
        DB_path=DB_path,
        data_source='FnGuide',
        train=True,
        data_date=data_date,
        F_day_type=5,
        T_day_type=5,
        Pred_Hrz=5,
        until_date=learn_DATE,
        stt_date=stt_DATE,
        cap_criterion=cap_criterion,
        transform=transform,
        country="KR"
    )
    # 테스트셋 로딩
    test_dataset = CustomDataset_all(
        image_data_path=image_path,
        DB_path=DB_path,
        data_source='FnGuide',
        train=False,
        data_date=data_date,
        F_day_type=5,
        T_day_type=5,
        Pred_Hrz=5,
        until_date=test_DATE,
        stt_date=learn_DATE + pd.DateOffset(days=1),
        cap_criterion=cap_criterion,
        transform=transform,
        country="KR"
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # [Step7] 반복 학습 (5회 수행)
    from sklearn.model_selection import StratifiedShuffleSplit
    from torch.utils.data import Subset, DataLoader

    ALL_RESULTS = []
    output_df = pd.DataFrame()

    for i in range(1, 3):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=(1 - TRAIN_RATIO), random_state=i*42)
        train_idx, val_idx = next(splitter.split(range(len(dataset)), dataset.labels))

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

        criterion = nn.CrossEntropyLoss()

        # save_model_path = f"C:/Users/flydc/jupyter_project/pjt1_deep_learning/models/KR_GoogLeNet/googlenet_model_{i}.pt"
        # save_history_path = f"C:/Users/flydc/jupyter_project/pjt1_deep_learning/models/KR_GoogLeNet/googlenet_history_{i}.pt"

        save_model_path = f"./models/KR_GoogleNet_SeNet/googlenet_model_{i}.pt"
        save_history_path = f"./models/KR_GoogleNet_SeNet/googlenet_history_{i}.pt"

        print(f"\n=========== [Iteration {i}] ===========")
        if os.path.exists(save_history_path):
            print(f'이미 학습된 모델이 존재합니다: {save_history_path}')
            pass
        else:
            print(f'모델 학습 시작: {save_model_path}')
            model = nn.DataParallel(GoogLeNet_CNN_5day(dr_rate=0.2, stt_chnl=1)).to(DEVICE)
            # model = nn.DataParallel(SENet_CNN_5day(dr_rate=0.5, stt_chnl=1)).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LR)

            train_acc, val_acc, train_epochs = Train_Nepoch_ES_AMP(
                model, train_loader, val_loader, criterion, DEVICE, optimizer,
                EPOCHS, BATCH_SIZE, TRAIN_RATIO, save_model_path, save_history_path, N_EPOCH_ES
            )

        # 모델 평가
        model = nn.DataParallel(GoogLeNet_CNN_5day(dr_rate=0.2, stt_chnl=1)).to(DEVICE)
        model.load_state_dict(torch.load(save_model_path))
        model_hist = torch.load(save_history_path)
        train_acc, val_acc, train_epochs = model_hist["acc_history"]['train'][-1], \
        model_hist["acc_history"]['val'][-1], len(model_hist["acc_history"]['train'])


        avg_loss, preds_tmp, codes, dates, returns, targets = eval_loop(test_loader, model, criterion, DEVICE)

        pred_probs = torch.nn.Softmax(dim=1)(preds_tmp)
        _1preds = torch.nn.Softmax(dim=1)(preds_tmp)[:, 1].cpu().numpy()

        preds = pred_probs.argmax(dim=1).cpu().numpy()

        acc = accuracy_score(targets, preds)
        prec = precision_score(targets, preds)
        rec = recall_score(targets, preds)
        f1 = f1_score(targets, preds)

        pred_result = inference_result_save(_1preds, codes, dates, returns, targets, train_epochs)
        pred_result = pred_result[pred_result['종목코드'].isin(CodeName.index)]
        pred_result['Prob_Positive_intRank_False'] = pred_result.groupby('date')['Prob_Positive'].rank(ascending=False)
        pred_result['Prob_Positive_pctRank_Flase'] = pred_result.groupby('date')['Prob_Positive'].rank(ascending=False, pct=True)
        pred_result['Prob_Positive_intRank_True'] = pred_result.groupby('date')['Prob_Positive'].rank(ascending=True)
        pred_result['Prob_Positive_pctRank_True'] = pred_result.groupby('date')['Prob_Positive'].rank(ascending=True, pct=True)

        # labels, predicts 비율 계산
        label_ratio = int(100 * (sum(targets) / len(targets)))
        pred_ratio = int(100 * (sum(preds) / len(preds)))

        print("=" * 30)
        print(f"[{i}] labels   : {label_ratio}%")
        print(f"[{i}] predicts : {pred_ratio}%")
        print(f"[{i}] Accuracy  : {acc*100:.2f}%")
        print(f"[{i}] Precision : {prec*100:.2f}%")
        print(f"[{i}] Recall    : {rec*100:.2f}%")
        print(f"[{i}] F1 Score  : {f1*100:.2f}%")

        ALL_RESULTS.append({
            "acc": acc, "prec": prec, "rec": rec, "f1": f1, "label_ratio": label_ratio, "pred_ratio": pred_ratio
        })
        tmp = pd.DataFrame({
            'iter': i,
            'acc_Train': [train_acc],
            'accVal': [val_acc],
            'acc_Test': [acc],
            'eps': [train_epochs],
            'avg_loss': [avg_loss],

            'prec': [prec],
            'rcll': [rec],
            'f1': [f1],

            'preds': [sum(preds) / len(preds)],
            'TOP5_AvgRtrn': pred_result.loc[pred_result['Prob_Positive_intRank_False'] <= 5, 'return'].mean(),
            'TOP10_AvgRtrn': pred_result.loc[pred_result['Prob_Positive_intRank_False'] <= 10, 'return'].mean(),
            'TOP30_AvgRtrn': pred_result.loc[pred_result['Prob_Positive_intRank_False'] <= 30, 'return'].mean(),
            'BTM5_AvgRtrn': pred_result.loc[pred_result['Prob_Positive_intRank_True'] <= 5, 'return'].mean(),
            'BTM10_AvgRtrn': pred_result.loc[pred_result['Prob_Positive_intRank_True'] <= 10, 'return'].mean(),
            'BTM30_AvgRtrn': pred_result.loc[pred_result['Prob_Positive_intRank_True'] <= 30, 'return'].mean(),

            'TOPQ1_Cnt': pred_result.loc[pred_result['Prob_Positive_pctRank_Flase'] <= 0.1, 'return'].groupby(
                'date').count().mean(),
            'TOPQ1_AvgRtrn': pred_result.loc[pred_result['Prob_Positive_pctRank_Flase'] <= 0.1, 'return'].mean(),
            'TOPQ2_AvgRtrn': pred_result.loc[pred_result['Prob_Positive_pctRank_Flase'] <= 0.2, 'return'].mean(),
            'TOPQ3_AvgRtrn': pred_result.loc[pred_result['Prob_Positive_pctRank_Flase'] <= 0.3, 'return'].mean(),
            'BTMQ1_AvgRtrn': pred_result.loc[pred_result['Prob_Positive_pctRank_True'] <= 0.1, 'return'].mean(),
            'BTMQ2_AvgRtrn': pred_result.loc[pred_result['Prob_Positive_pctRank_True'] <= 0.2, 'return'].mean(),
            'BTMQ3_AvgRtrn': pred_result.loc[pred_result['Prob_Positive_pctRank_True'] <= 0.3, 'return'].mean(),

            'THR50_Cnt': pred_result.loc[pred_result['Prob_Positive'] >= 0.50, 'return'].groupby('date').count().mean(),
            'THR55_Cnt': pred_result.loc[pred_result['Prob_Positive'] >= 0.55, 'return'].groupby('date').count().mean(),
            'THR60_Cnt': pred_result.loc[pred_result['Prob_Positive'] >= 0.60, 'return'].groupby('date').count().mean(),
            'THR50_AvgRtrn': pred_result.loc[pred_result['Prob_Positive'] >= 0.50, 'return'].mean(),
            'THR55_AvgRtrn': pred_result.loc[pred_result['Prob_Positive'] >= 0.55, 'return'].mean(),
            'THR60_AvgRtrn': pred_result.loc[pred_result['Prob_Positive'] >= 0.60, 'return'].mean(),
        })
        output_df = pd.concat([output_df, tmp], ignore_index=True)
        print(output_df)



    # [마무리] 평균 성능 출력
    print("\n\n====== 평균 성능 ======")
    mean_acc = sum([x['acc'] for x in ALL_RESULTS]) / len(ALL_RESULTS)
    mean_prec = sum([x['prec'] for x in ALL_RESULTS]) / len(ALL_RESULTS)
    mean_rec = sum([x['rec'] for x in ALL_RESULTS]) / len(ALL_RESULTS)
    mean_f1 = sum([x['f1'] for x in ALL_RESULTS]) / len(ALL_RESULTS)
    mean_label_ratio = sum([x['label_ratio'] for x in ALL_RESULTS]) / len(ALL_RESULTS)
    mean_pred_ratio = sum([x['pred_ratio'] for x in ALL_RESULTS]) / len(ALL_RESULTS)

    print(f"Mean labels   : {mean_label_ratio:.2f}%")
    print(f"Mean predicts : {mean_pred_ratio:.2f}%")
    print(f"Mean Accuracy  : {mean_acc*100:.2f}%")
    print(f"Mean Precision : {mean_prec*100:.2f}%")
    print(f"Mean Recall    : {mean_rec*100:.2f}%")
    print(f"Mean F1 Score  : {mean_f1*100:.2f}%")
    output_df.to_excel(f'./googlenet_results.xlsx', index=False)


