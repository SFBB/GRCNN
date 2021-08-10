import torch.nn as nn
import datetime

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import torch
import torch.nn.functional as F
# from A_softmax import A_softmax
import torch.nn as nn
# from WeightedCrossEntropyLoss import WeightedCrossEntropyLoss
import torch.optim as optim
import numpy as np
# from dataio import dataio
import ast
# from cal_eer_different_types import spoof19_cal
# from run_tDCF import run_tDCF
# from CLDNN import Model
# from focalloss import FocalLoss
# from compute_eer import cal_script
# from ghm_loss import GHMC_Loss
from tqdm import tqdm
import argparse
import sys
from dataloader import TorchDataset, custom_collate
from torch.utils.data import DataLoader
from GRCNN import GRCNNs

# from compute_eer import gen_tDCF

os.chdir(os.path.split(os.path.realpath(__file__))[0])



parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in data-loading')
parser.add_argument('--nepoch', type=int, default=150, help='max epoch')
parser.add_argument('--sample_rate', type=int, default=16000, help='sample_rate')

parser.add_argument('--load_model_from', type=str, default=None, help='load_model_from')
parser.add_argument('--model_ID', type=int, default=-1, help='eval ID')
parser.add_argument('--task', type=str, default='train', help='task', choices=['train', 'test'])

parser.add_argument('--loss', type=str, default='CE', help='loss function', choices=['CE', 'FocalLoss'])

parser.add_argument('--volume_rate', type=float, default=1, help='volume rate for music')
parser.add_argument('--norm_wav', type=ast.literal_eval, dest='norm_wav', help='set wav [-2^15, 2^15] when read', default=False)

parser.add_argument('--addmusic', type=ast.literal_eval, dest='addmusic', help='addmusic flag', default=False)
parser.add_argument('--music_path', type=str, default='musdb18_accompaniment', help='useful only if addmusic is true')

parser.add_argument('--truncate', type=ast.literal_eval, dest='truncate', help='truncate', default=True)

parser.add_argument('--dim', type=int, default=1, help='dim')
parser.add_argument('--length', type=int, default=96000, help='length')


# for noise
parser.add_argument('--noise_rate', type=float, default=1.0, help='noise rate')
parser.add_argument('--noise', type=str, default=None, help='noise type', choices=['gauss',])


parser.add_argument('--access_type', type=str, default='LA', help='access_type')
parser.add_argument('--pathToData', type=str, default='1dim', help='pathToData')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')

args = parser.parse_args()
print('args: ', args)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("sadasdasd")
print(device)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

if args.task == 'train':
    if args.load_model_from:
        model_ID = int(args.load_model_from.split('/')[0].split('_')[-1])
        print('load from model_ID: ', model_ID)
    else:
        # model_ID = 0 if debug else np.random.randint(100000000)
        model_ID = np.random.randint(100000000)
        print('model_ID: ', model_ID)
        now = datetime.datetime.now().strftime("%Y_%m_%d-%H%M")
        os.makedirs('models/{}_models_{}_{}'.format(args.access_type, model_ID, now), exist_ok=True)
        os.makedirs('results/{}_results_{}_{}'.format(args.access_type, model_ID, now), exist_ok=True)
        model_dir = 'models/{}_models_{}_{}'.format(args.access_type, model_ID, now)
    args.model_ID = model_ID
elif args.task == 'eval':
    if args.model_ID == -1:
        raise ValueError('eval id required.')
    model_ID = args.model_ID
    print('eval ID: ', args.model_ID)




# sys.stdout.flush()



# def transform_lables_for_CE(labels):
#     result = []
#     for label in labels:
        

def train():
    path = "/media/ssd1T/antispoof/2019/LA"
    train_data = TorchDataset(data_list=path+"/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", data_dir="/media/ssd1T/anzhe/GRCNN/", num_classes=20, repeat=None)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=True)

    net = GRCNNs(20).cuda()

    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    start_epoch = 0
    count = 0
    last_loss = 100
    # with torch.no_grad():
    for epoch in range(start_epoch, args.nepoch):
        net.train()
        running_loss = 0.0
        t = tqdm(range(train_data.get_len()//args.batch_size))
        train_loader_iter = iter(train_loader) # 每次这个iter创建时，trainloader会自动shuffle
        for step in t:
            (batch_flacs_MGD, batch_flacs_STFT, batch_labels) = next(train_loader_iter)
            # print(step)
            # print(batch_flacs, batch_labels)
            # print(len(batch_flacs), len(batch_labels))
            # print(batch_flacs_MGD[0].shape)
            # print(batch_flacs_STFT[0].shape)
            # print(batch_labels[0][1].shape)
            # pass
            # print(len(batch_flacs), batch_flacs[0].shape)
            # print(np.concatenate(batch_flacs, axis=0).shape)
            # batch_flacs = torch.from_numpy(np.concatenate(batch_flacs, axis=0)).to(device)
            # batch_labels = torch.from_numpy(np.concatenate(batch_labels, axis=0)).to(device)
            optimizer.zero_grad()

            preditcted_labels = net(batch_flacs_MGD, batch_flacs_STFT, device)
            # print(preditcted_labels.shape)
            # print(batch_labels.shape)

            loss = criterion(preditcted_labels, torch.as_tensor(batch_labels).cuda())
            del preditcted_labels
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # t.update(step)
            # if running_loss / (step + 1) >= last_loss:
            #     count += 1
            # else:
            #     last_loss = running_loss / (step + 1)
            # if count > 5:
            #     break
            t.set_description('[%02d] Loss: %.5f' % (epoch + 1, running_loss / (step + 1)))
            if step >= train_data.get_len()//args.batch_size:
                break
        
        torch.save(net.state_dict(), model_dir+'/params_%d.pkl' % (epoch + 1))
        if running_loss / (train_data.get_len()//16) >= last_loss:
            count += 1
        else:
            last_loss = running_loss / (train_data.get_len()//args.batch_size)
            count = 0
        print('\n[%d] loss: %.5f' % (epoch + 1, running_loss / (train_data.get_len()//args.batch_size)))
        if count > 5:
            break


if __name__ == "__main__":
    if args.task == "train":
        train() # 具体参数在这个train中调节