import numpy as np
import torch
import torch.nn as nn
import datetime
import time
import argparse
import math
from model.dataload import split_and_norm_data_time
from model.STGATN import STGATN

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='STGATN', help='dataset name')
parser.add_argument('--cuda', type=int, default=0,  help='which gpu card used')
parser.add_argument('--P', type=int, default=12, help='history steps')
parser.add_argument('--Q', type=int, default=6, help='prediction steps')
parser.add_argument('--N', type=int, default=12, help='stations')
parser.add_argument('--d', type=int, default=8, help='dims of each head attention outputs')
parser.add_argument('--L', type=int, default=1, help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=4, help='number of attention heads')
parser.add_argument('--train_ratio', type=float, default=0.5, help='training set [default : 0.8]')
parser.add_argument('--val_ratio', type=float, default=0.25, help='validation set [default : 0.2]')
parser.add_argument('--test_ratio', type=float, default=0.25, help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--max_epoch', type=int, default=50, help='epoch to run')
parser.add_argument('--patience', type=int, default=10, help='patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10, help='decay epoch')
parser.add_argument('--data_file', default='data/train.npz', help='train file')
parser.add_argument('--model_file', default='PEMS', help='save the model to disk')
parser.add_argument('--log_file', default='log(PEMS)', help='log file')
args = parser.parse_args()

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
log = open(args.log_file, 'w')

seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
np.random.seed(seed)
# random.seed(seed) 

def res(model, testX, testDoW, testH, testY, valXAll, mean, std):
    model.eval()  # 评估模式, 这会关闭dropout
    num_val = testX.shape[0]
    SE = np.array([i for i in range(args.N)],dtype=np.int64)
    pred = []
    label = []
    num_batch = math.ceil(num_val / args.batch_size)
    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
                X = torch.from_numpy(testX[start_idx: end_idx]).float().to(device)
                DOW = torch.from_numpy(testDoW[start_idx: end_idx]).float().to(device)
                H = torch.from_numpy(testH[start_idx: end_idx]).float().to(device)
                Y = testY[start_idx: end_idx]
                XAll = torch.from_numpy(valXAll[start_idx: end_idx]).float().to(device)
                y_hat = model(X = X, SE = SE, TE = [DOW, H], XAll=XAll)
                pred.append(y_hat.cpu().numpy())
                label.append(Y)
    pred = np.concatenate(pred, axis=0)
    label = np.concatenate(label, axis=0)
    mae, rmse, mape = metric(pred * std + mean, label)
    return mae, rmse, mape, pred * std + mean, label


def train(model, trainX, trainDoW, trainH, trainY, trainXAll, valX, valDoW, valH, valY, valXAll, testX, testDoW, testH, testY, testXAll, mean, std):
    TotalX = np.concatenate([trainX, valX],axis=0)
    TotalDoW = np.concatenate([trainDoW, valDoW], axis=0)
    TotalH = np.concatenate([trainH, valH], axis=0)
    TotalY = np.concatenate([trainY, valY], axis=0)
    TotalXAll = np.concatenate([trainXAll, valXAll], axis=0)

    num_total = TotalX.shape[0]
    permutation = np.random.permutation(num_total)
    TotalX = TotalX[permutation]
    TotalY = TotalY[permutation]
    TotalDoW = TotalDoW[permutation]
    TotalH = TotalH[permutation]
    TotalXAll = TotalXAll[permutation]

    train_high = round(2/3 * TotalX.shape[0])
    trainX, trainY, trainDoW, trainH, trainXAll = TotalX[:train_high], TotalY[:train_high], TotalDoW[:train_high], TotalH[:train_high], TotalXAll[:train_high]
    valX, valY, valDoW, valH, valXAll = TotalX[train_high:], TotalY[train_high:], TotalDoW[train_high:], TotalH[train_high:], TotalXAll[train_high:]
    del TotalX, TotalDoW, TotalH, TotalY, TotalXAll
    num_train = trainX.shape[0]
    min_loss = 10000000.0
    SE = np.array([i for i in range(args.N)],dtype=np.int32)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.decay_epoch,
                                                    gamma=0.9)
    patience=0
    iteration = 0
    start_time = datetime.datetime.now()
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        permutation = np.random.permutation(num_train)
        trainX = trainX[permutation]
        trainY = trainY[permutation]
        trainDoW = trainDoW[permutation]
        trainH = trainH[permutation]
        trainXAll = trainXAll[permutation]
        num_batch = math.ceil(num_train / args.batch_size)
        for batch_idx in range(num_batch):
            iteration+=1
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
            X = torch.from_numpy(trainX[start_idx: end_idx]).float().to(device)
            DOW = torch.from_numpy(trainDoW[start_idx: end_idx]).float().to(device)
            H = torch.from_numpy(trainH[start_idx: end_idx]).float().to(device)
            Y = torch.from_numpy(trainY[start_idx: end_idx]).float().to(device)
            XAll = torch.from_numpy(trainXAll[start_idx: end_idx]).float().to(device)
            optimizer.zero_grad()
            y_hat = model(X = X, SE = SE, TE = [DOW, H], XAll=XAll)
            loss = _compute_loss(Y, y_hat * std + mean)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_l_sum += loss.cpu().item()
            n += Y.shape[0]
            batch_count += 1

            if iteration==100:
                end_time = datetime.datetime.now()
                total_time = end_time - start_time
                print("Total training times is : %f" % total_time.total_seconds())
        log_string(log, 'in the training step, epoch %d, lr %.6f, loss %.4f, time %.1f sec' % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
        mae, rmse, mape, _, _ = res(model, valX, valDoW, valH, valY, valXAll, mean, std)
        lr_scheduler.step()
        if patience>args.patience: break
        else:patience += 1
        if mae < min_loss:
            patience = 0
            log_string(log, 'in the %dth epoch, the validate average loss value is : %.3f' % (epoch+1, mae))
            testmae, testrmse, testmape, _, _ = res(model, testX, testDoW, testH, testY, testXAll, mean, std)
            log_string(log, 'in the %dth epoch, the test average loss value is : %.3f' % (epoch+1, testmae))
            min_loss = mae
            torch.save(model, args.model_file)

def test(testX, testDoW, testH, testY, testXAll, mean, std):
    model = torch.load(args.model_file)
    mae, rmse, mape, pred, label = res(model, testX, testDoW, testH, testY, testXAll, mean, std)
    log_string(log, 'in the test phase,  mae: %.4f, rmse: %.4f, mape: %.6f' % (mae, rmse, mape))
    return pred, label

def _compute_loss(y_true, y_predicted):
    return masked_mae(y_predicted, y_true, 0.0)

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        # mask = (label >=800) & (label <= 1000)
        pred = pred[[i for i in range(18, pred.shape[0], 24)]]
        label = label[[i for i in range(18, label.shape[0], 24)]]

        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)

        mean_obs = np.mean(label * mask)   
        numerator = np.sum((pred - label) ** 2 * mask)   
        denominator = np.sum((np.abs(pred - mean_obs) + np.abs(label - mean_obs)) ** 2 * mask)
        ia = 1 - (numerator / denominator) 

        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, ia

if __name__ == '__main__':
    log_string(log, "model constructed begin....")
    model = STGATN(args=args, bn_decay=0.1, device=device).to(device)
    log_string(log, "model constructed end....")

    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))

    log_string(log, "loading data....")
    trainX, trainDoW, trainH, trainY, trainXAll, valX, valDoW, valH, valY, valXAll, testX, testDoW, testH, testY, testXAll, mean, std = split_and_norm_data_time(
        args)
    log_string(log, f'trainX: {trainX.shape}\t\t trainY: {trainY.shape}')
    log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
    log_string(log, f'testX:   {testX.shape}\t\ttestY:   {testY.shape}')
    log_string(log, f'mean:   {mean:.4f}\t\tstd:   {std:.4f}')
    log_string(log, 'data loaded!')
    log_string(log, "loading end....")

    log_string(log, "train begin....")
    # train(model,trainX, trainDoW, trainH, trainY, trainXAll, valX, valDoW, valH, valY, valXAll, testX, testDoW, testH, testY, testXAll, mean, std)
    log_string(log, "train end....")

    start_time = datetime.datetime.now()
    predicted, observed = test(testX, testDoW, testH, testY, testXAll, mean, std)
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print("Total validating times is : %f" % total_time.total_seconds())

    print(predicted.shape, observed.shape)
    print('                MAE\t\tRMSE\t\tIA')
    for i in range(args.Q):
        mae, rmse, mape = metric(predicted[:,:i+1], observed[:,:i+1])
        print('step: %02d         %.3f\t\t%.3f\t\t%.4f' % (i + 1, mae, rmse, mape))

    # for i in range(predicted.shape[0]):
    #     print(predicted[i], observed[i])
    np.savez_compressed('STGATN', **{'prediction': predicted, 'truth': observed})
    # mae, rmse, mape = metric(predicted, observed)
    # log_string(log, 'final results,  average mae: %.4f, rmse: %.4f, mape: %.6f' % (mae, rmse, mape))