# -- coding: utf-8 --
import numpy as np
import datetime
def seq2instance(data, P, Q, low=0, high=100, sites = 108, type='train'):
    '''
    :param data:
    :param P:
    :param Q:
    :param low_index:
    :param high_index:
    :param granularity:
    :param sites:
    :param type:

    :return: trainX,         trainDoW,   trainH,      trainL,        trainXAll
             (17334, 24, 12) (17334, 30) (17334, 30) (17334, 6, 12) (17334, 30, 12)
    '''
    X, DoW, H, L, XAll = [],[],[],[],[]
    total_week_len = 24 * 7

    while low + P + Q < high:
        L.append(np.reshape(data[low + P: (low + P + Q), :, 5], [1, Q, sites]))
        X.append(np.reshape(data[low: (low + P), :, [5, 15]],[1, P, sites, -1]))
        date = data[low: (low + P + Q), -1, 1:4]
        H.append(np.reshape(data[low: (low + P + Q), -1, 4], [1, P + Q]))
        DoW.append(np.reshape([datetime.date(int(char[0]), int(char[1]), int(char[2])).weekday() for char in date],[1, P+Q]))
        XAll.append(np.reshape(data[(low- total_week_len) : (low - total_week_len + P + Q), :, [5, 15]],[1, P+Q, sites, -1]))

        if type =='train':
            low += 1
        else:
            low += 1

    return np.concatenate(X,axis=0), \
           np.concatenate(DoW,axis=0), \
           np.concatenate(H,axis=0), \
           np.concatenate(L,axis=0), \
           np.concatenate(XAll,axis=0)


def split_and_norm_data_time(args):
    # dataset
    data = np.load(args.data_file, allow_pickle=True)['data']
    # data = np.delete(data, 8, axis=2)
    print(data.shape)
    # train/val/test
    samples, sites, features = data.shape
    total_samples = samples

    train_low = 24 * 7
    val_low = round(args.train_ratio * total_samples)
    test_low = round((args.train_ratio + args.val_ratio) * total_samples)

    # X, Y, day of week, hour, label, all X
    trainX, trainDoW, trainH, trainL, trainXAll = seq2instance(data,
                                                               args.P,
                                                               args.Q,
                                                               low=train_low,
                                                               high=val_low,
                                                               sites=args.N,
                                                               type='train')
    print('training dataset has been loaded!')
    valX, valDoW, valH, valL, valXAll = seq2instance(data,
                                                     args.P,
                                                     args.Q,
                                                     low=val_low,
                                                     high=test_low,
                                                     sites=args.N,
                                                     type='validation')
    print('validation dataset has been loaded!')
    testX, testDoW, testH, testL, testXAll = seq2instance(data,
                                                          args.P,
                                                          args.Q,
                                                          low=test_low,
                                                          high=total_samples,
                                                          sites=args.N,
                                                          type='test')
    print('testing dataset has been loaded!')
    # normalization
    
    mean, std = (np.array([np.mean(data[:, :, 5]), np.mean(data[:, :, 15])], dtype=float),
                 np.array([np.std(data[:, :, 5]), np.std(data[:, :, 15])],dtype=float))
    trainX, trainXAll = (trainX  - mean) / (std), (trainXAll  - mean) / (std)
    valX, valXAll = (valX  - mean) / (std), (valXAll  - mean) / (std)
    testX, testXAll = (testX  - mean) / (std), (testXAll  - mean) / (std)
    
    print(mean, std)
    return (trainX[:,:,:,0:1], trainDoW, trainH, trainL, trainXAll[:,:,:,0:1],
            valX[:,:,:,0:1], valDoW, valH, valL, valXAll[:,:,:,0:1],
            testX[:,:,:,0:1], testDoW, testH, testL, testXAll[:,:,:,0:1],
            mean[0], std[0])


'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='STGATN', help='dataset name')
parser.add_argument('--cuda', type=int, default=0,  help='which gpu card used')
parser.add_argument('--P', type=int, default=24, help='history steps')
parser.add_argument('--Q', type=int, default=6, help='prediction steps')
parser.add_argument('--N', type=int, default=12, help='stations')
parser.add_argument('--d', type=int, default=64, help='dims of outputs')
parser.add_argument('--L', type=int, default=1, help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8, help='number of attention heads')
# parser.add_argument('--d', type=int, default=8, help='dims of each head attention outputs')
parser.add_argument('--train_ratio', type=float, default=0.5, help='training set [default : 0.8]')
parser.add_argument('--val_ratio', type=float, default=0.25, help='validation set [default : 0.2]')
parser.add_argument('--test_ratio', type=float, default=0.25, help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--max_epoch', type=int, default=100, help='epoch to run')
parser.add_argument('--patience', type=int, default=10, help='patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10, help='decay epoch')
parser.add_argument('--data_file', default='/Users/zouguojian/Pollution_predic/data/train.npz', help='coin file')
parser.add_argument('--model_file', default='PEMS', help='save the model to disk')
parser.add_argument('--log_file', default='log(PEMS)', help='log file')
args = parser.parse_args()

trainX, trainDoW, trainH, trainL, trainXAll, valX, valDoW, valH, valL, valXAll, testX, testDoW, testH, testL, testXAll, mean, std = split_and_norm_data_time(args)
print(trainX.shape, trainDoW.shape, trainH.shape, trainL.shape, trainXAll.shape, mean, std)
'''



'''
'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM'
'''