import os
import math
import numpy as np

from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
def cal_eer(score_true, score_false):
    """ 计算EER
    Args:
    scores_true: 正样例的分数列表
    scores_false: 负样例的分数列表
    Return:
    (EER, threshold)
    """
    
    fpr, tpr, thresholds = metrics.roc_curve([1]*len(score_true)+[0]*len(score_false), score_true+score_false, pos_label=1)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

def compute_eer(scores, labels, with_threshold=False):
    eps = 1e-300
    if isinstance(labels[0], str):
        labels = [1 if s == 'target' else 0 for s in labels]
    if isinstance(labels, list):
        labels = np.asarray(labels)
    if isinstance(scores, list):
        scores = np.asarray(scores)
    if isinstance(labels, list):
        labels = np.asarray(labels)
    assert len(scores) == len(labels)
    
    # print('EER by sklearn')
    # cal_eer(scores[labels == 1], scores[labels == 0])
    # print('---end---')

    I = np.argsort(scores)
    x = labels[I]

    FNR = np.cumsum(x == 1) / (np.sum(x == 1) + eps)
    TNR = np.cumsum(x == 0) / (np.sum(x == 0) + eps)
    FPR = 1 - TNR
    # TP = 1 - FN
    # FNR = FN / (TP + FN + eps)
    # FPR = FP / (TN + FP + eps)
    difs = FNR - FPR
    idx1 = np.searchsorted(difs, -1e-20) - 1
    idx2 = idx1 + 1
    x = [FNR[idx1], FPR[idx1]]
    y = [FNR[idx2], FPR[idx2]]

    
    a = ( x[0] - x[1] ) / ( y[1] - x[1] - y[0] + x[0])
    eer = 100 * ( x[0] + a * ( y[0] - x[0] ) )
    
    if False:
        threshold = 0.5
        FNrate = np.sum((labels == 1) & (scores < threshold)) / (np.sum(labels == 1) + eps)
        FTrate = np.sum((labels == 0) & (scores > threshold)) / (np.sum(labels == 0) + eps)
        print('FPR FNR (thres 0.5): %.2f%% %.2f%%' % (FTrate * 100, FNrate * 100))

    if with_threshold:
        sc =  np.sort(scores)
        threshold = (sc[idx1] + sc[idx2]) / 2
        # FNrate = np.sum((labels == 1) & (scores < threshold)) / (np.sum(labels == 1) + eps)
        # FTrate = np.sum((labels == 0) & (scores > threshold)) / (np.sum(labels == 0) + eps)
        # print('FPR FNR (thres of EER): ', FTrate, FNrate)
        return eer, threshold
    else:
        return eer



def cmp(lab, res):
    f1 = open(lab)
    f2 = open(res)
    tdcf_file = open(res.replace('.txt', '_tDCF.txt'), 'w')

    x = []
    y = []
    cnt = 0
    correct = 0
    l1 = f1.readline().strip()
    l2 = f2.readline().strip()

    while True:
        if not l1 or not l2:
            break
        
        if l1.split(' ')[1] != l2.split(' ')[0].split('.')[0]:
            raise ValueError('error')
            # l1 = f1.readline().strip()
            # continue
        
        # print(l2.split(' ')[1] + '\t' + l1.split(' ')[-1])
        
        a = float(l2.split(' ')[1])
        b = float(l2.split(' ')[2])
        # assert abs(a + b - 1) < 0.0001
        # x.append(math.exp(a) / (math.exp(a) + math.exp(b)))
        x.append(a)
        tdcf_file.write('%s %s %s %s\n' % (l1.split(' ')[1], l1.split(' ')[3], l1.split(' ')[4], str(a)))
        
        # x.append(a-b)
        # thres = 0.5
        if l1.split(' ')[-1].strip() == 'bonafide':
            y.append(1)
            if a > b:
                correct += 1
        elif l1.split(' ')[-1].strip() == 'spoof':
            y.append(0)
            if a < b:
                correct += 1
        else:
            raise ValueError('unknown type')

        cnt += 1
        
        l1 = f1.readline().strip()
        l2 = f2.readline().strip()

    # print(x)
    # print(y)
    # x = softmax(np.asarray(x))
    
    x = np.asarray(x)
    y = np.asarray(y)
    # eps = 1e-300
    # FNR = np.sum((y == 1) & (x < 0.5)) / (np.sum(y == 1) + eps)
    # FPR = np.sum((y == 0) & (x > 0.5)) / (np.sum(y == 0) + eps)
    # print('FNR: %.2f%%' % (FNR * 100))
    # print('FPR: %.2f%%' % (FPR * 100))

    # print('IR: %.2f%%'% (correct / cnt * 100))
    # print(lab.split('/')[-1], 'num: ', cnt)
    tdcf_file.close()
    if 'eval' in lab:
        print('EVAL EER: %.2f%% thres: %.5f num: %d\n' % (*compute_eer(x, y, True), cnt))
    else:
        print(' dev EER: %.2f%% thres: %.5f num: %d' % (*compute_eer(x, y, True), cnt))

access_type = 'LA'
pathToASVspoof2019Data = '/media/ssd512/gavin/ASVspoof2019'

pathToDatabase = os.path.join(pathToASVspoof2019Data, access_type)

trainProtocolFile = os.path.join(pathToDatabase, 'ASVspoof2019_' + access_type + '_cm_protocols', 'ASVspoof2019.' + access_type + '.cm.train.trn.txt')


devProtocolFile = os.path.join(pathToDatabase, 'ASVspoof2019_' + access_type + '_cm_protocols', 'ASVspoof2019.' + access_type + '.cm.dev.trl.txt')

evalProtocolFile = os.path.join(pathToDatabase, 'ASVspoof2019_' + access_type + '_cm_protocols', 'ASVspoof2019.' + access_type + '.cm.eval.trl.txt')


def gen_tDCF(randID):
    if os.path.exists('{}_results_{}/dev.txt'.format(access_type, randID)):
        cmp(devProtocolFile, '{}_results_{}/dev.txt'.format(access_type, randID))

    if os.path.exists('{}_results_{}/eval.txt'.format(access_type, randID)):
        cmp(evalProtocolFile, '{}_results_{}/eval.txt'.format(access_type, randID))





# def compute_FPR_FNR(scores, labels, threshold=0):
#     if USE_GPU:
#         import cupy as cp
#     import numpy as np
#     if USE_GPU and isinstance(scores, cp.ndarray):
#         scores = scores.get()
#     if USE_GPU and isinstance(labels, cp.ndarray):
#         labels = labels.get()
    
#     eps = 1e-300
#     if isinstance(labels[0], str):
#         labels = [1 if s == 'target' else 0 for s in labels]
#     if isinstance(labels, list):
#         labels = np.asarray(labels)
#     if isinstance(scores, list):
#         scores = np.asarray(scores)
#     if isinstance(labels, list):
#         labels = np.asarray(labels)
#     assert len(scores) == len(labels)

#     FNrate = np.sum((labels == 1) & (scores < threshold)) / (np.sum(labels == 1) + eps)
#     FTrate = np.sum((labels == 0) & (scores > threshold)) / (np.sum(labels == 0) + eps)
#     return FTrate, FNrate

