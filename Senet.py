import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torch import nn, optim

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

os.makedirs("C:/Users/flydc/jupyter_project/pjt1_deep_learning/models/KR_SENet", exist_ok=True)

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
    image_path = "C:/Users/flydc/jupyter_project/pjt1_deep_learning/h5_image"
    DB_path = "C:/Users/flydc/jupyter_project/pjt1_deep_learning/hd5_image"
    data_date = "20250527"
    cap_criterion = 0.0

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

    # [Step7] 반복 학습 (5회 수행)
    from sklearn.model_selection import StratifiedShuffleSplit
    from torch.utils.data import Subset, DataLoader

    ALL_RESULTS = []
    for i in range(1, 3):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=(1 - TRAIN_RATIO), random_state=i*42)
        train_idx, val_idx = next(splitter.split(range(len(dataset)), dataset.labels))

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

        model = nn.DataParallel(SENet_CNN_5day(dr_rate=0.2, stt_chnl=1)).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        criterion = nn.CrossEntropyLoss()

        save_model_path = f"C:/Users/flydc/jupyter_project/pjt1_deep_learning/models/KR_SENet/senet_model_{i}.pt"
        save_history_path = f"C:/Users/flydc/jupyter_project/pjt1_deep_learning/models/KR_SENet/senet_history_{i}.pt"

        print(f"\n=========== [Iteration {i}] ===========")
        train_acc, val_acc, train_epochs = Train_Nepoch_ES_AMP(
            model, train_loader, val_loader, criterion, DEVICE, optimizer,
            EPOCHS, BATCH_SIZE, TRAIN_RATIO, save_model_path, save_history_path, N_EPOCH_ES
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

        # 모델 평가
        model.load_state_dict(torch.load(save_model_path))
        avg_loss, preds_tmp, codes, dates, returns, targets = eval_loop(test_loader, model, criterion, DEVICE)

        pred_probs = torch.nn.Softmax(dim=1)(preds_tmp)
        preds = pred_probs.argmax(dim=1).cpu().numpy()

        acc = accuracy_score(targets, preds)
        prec = precision_score(targets, preds)
        rec = recall_score(targets, preds)
        f1 = f1_score(targets, preds)
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

