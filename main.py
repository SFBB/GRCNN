import torch.nn as nn
import datetime

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"]=""
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
from dataloader import TorchDataset, custom_collate, custom_collate_for_dev
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
parser.add_argument('--task', type=str, default='train', help='task', choices=['train', 'dev', 'test', 'dev_and_eval'])

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
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser.add_argument('--cuda', type=str2bool, default=True, help='cuda or not')

args = parser.parse_args()
print('args: ', args)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("sadasdasd")
print(device)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

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
def cal_accuracy(scores, labels, all, wrong):
    predicts = torch.max(scores, 1).indices
    for index, predict in enumerate(predicts):
        if predict != labels[index]:
            wrong += 1
        all += 1
    # print(wrong, all)
    return 1-wrong/all, all, wrong

def eval(net):
    path = "/media/ssd1T/antispoof/2019/LA"
    result_path = "cm.eval.scores.txt"
    eval_data = TorchDataset(data_list=path+"/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", data_dir="/media/ssd1T/anzhe/GRCNN/", task="eval", num_classes=20, repeat=None)
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, collate_fn=custom_collate_for_dev, shuffle=False)
    with open(result_path, "w+") as file:
        # file_ = open(result_path+".labels", "w+")
        t = tqdm(range(eval_data.get_len()))
        eval_loader_iter = iter(eval_loader)
        net.eval()
        softmax = torch.nn.Softmax(dim=1)

        all = 0
        wrong = 0
        with torch.no_grad():
            for step in t:
                (batch_flacs_MGD, batch_flacs_STFT, batch_ids, batch_names, batch_types, batch_genders, batch_labels) = next(eval_loader_iter)
                # model.to("cpu")
                # print(args.cuda)
                batch_flacs_MGD = torch.from_numpy(batch_flacs_MGD).float()
                batch_flacs_STFT = torch.from_numpy(batch_flacs_STFT).float()
                if args.cuda:
                    batch_flacs_MGD = batch_flacs_MGD.cuda()
                    batch_flacs_STFT = batch_flacs_STFT.cuda()

                preditcted_labels = net(batch_flacs_MGD, batch_flacs_STFT, args.cuda)
                scores = softmax(preditcted_labels)
                # scores = preditcted_labels
                # for index, flac in enumerate(batch_names):
                    # file.write("{} {:.10f} {:.10f}\n".format(flac, scores[index][0], scores[index][1]))
                file.write("{} {} - {} {} {:.10f}\n".format(batch_ids[0], batch_names[0], batch_types[0], batch_genders[0], scores[0][0]))

                    # file_.write("{} {:.10f} {:.10f} {}\n".format(flac, scores[index][0], scores[index][1], batch_labels[index]))

                accuracy, all, wrong = cal_accuracy(scores, batch_labels, all, wrong)
                t.set_description('[%02d] Accuracy: %.5f' % (step + 1, accuracy))
            print("Finished! Accuracy on eval is %.5f" % accuracy)
                # scores = scores[:, 0]
                # print(next(model.parameters()).is_cuda)
                # exit(0)
                # else:
                    # model.to(torch.device("cpu"))
                # output = model(batch, batch_, args.cuda)
                # print(torch.max(output, 1))
                # print(label)

def dev(net):
    path = "/media/ssd1T/antispoof/2019/LA"
    result_path = "cm.dev.scores.txt"
    dev_data = TorchDataset(data_list=path+"/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt", data_dir="/media/ssd1T/anzhe/GRCNN/", task="dev", num_classes=20, repeat=None)
    dev_loader = DataLoader(dataset=dev_data, batch_size=1, collate_fn=custom_collate_for_dev, shuffle=False)
    with open(result_path, "w+") as file:
        # file_ = open(result_path+".labels", "w+")
        t = tqdm(range(dev_data.get_len()))
        dev_loader_iter = iter(dev_loader)
        # net = load_model()
        # if args.cuda:
        #     net.cuda()
        net.eval()
        softmax = torch.nn.Softmax(dim=1)

        all = 0
        wrong = 0
        with torch.no_grad():
            for step in t:
                (batch_flacs_MGD, batch_flacs_STFT, batch_ids, batch_names, batch_types, batch_genders, batch_labels) = next(dev_loader_iter)
                # model.to("cpu")
                # print(args.cuda)
                batch_flacs_MGD = torch.from_numpy(batch_flacs_MGD).float()
                batch_flacs_STFT = torch.from_numpy(batch_flacs_STFT).float()
                if args.cuda:
                    batch_flacs_MGD = batch_flacs_MGD.cuda()
                    batch_flacs_STFT = batch_flacs_STFT.cuda()

                preditcted_labels = net(batch_flacs_MGD, batch_flacs_STFT, args.cuda)
                scores = softmax(preditcted_labels)
                # scores = preditcted_labels
                # for index, flac in enumerate(batch_names):
                file.write("{} {} - {} {} {:.10f}\n".format(batch_ids[0], batch_names[0], batch_types[0], batch_genders[0], scores[0][0]))
                    # file_.write("{} {} {:.10f} {:.10f} {}\n".format(flac, scores[index][0], scores[index][1], batch_labels[index]))

                accuracy, all, wrong = cal_accuracy(scores, batch_labels, all, wrong)
                t.set_description('[%02d] Accuracy: %.5f' % (step + 1, accuracy))
            print("Finished! Accuracy on dev is %.5f" % accuracy)
                # scores = scores[:, 0]
                # print(next(model.parameters()).is_cuda)
                # exit(0)
                # else:
                    # model.to(torch.device("cpu"))
                # output = model(batch, batch_, args.cuda)
                # print(torch.max(output, 1))
                # print(label)

def dev_and_eval():
    net = load_model()
    if args.cuda:
        net.cuda()
    net.eval()
    dev(net)
    eval(net)

def load_model():
    model_param_path = '{}_models_{}'.format(args.access_type, args.model_ID)
    models = os.listdir("models/")
    models_ = []
    print(model_param_path)
    for model in models:
        if model_param_path in model:
            models_.append(model)
    print(models_)
    r = input("Those models fit in your model id, which one do you want to load?")
    model_param_path = os.path.join("models", models_[int(r)])
    params = os.listdir(model_param_path)
    params_ = []
    for param in params:
        params_.append(int(param.split("_")[1].split(".")[0]))
    params_.sort()
    print(params_)
    r = input("Those versions this model has, which one do you want?")
    param = int(r)
    model = GRCNNs(2)
    model.load_state_dict(torch.load(os.path.join(model_param_path, "params_%d.pkl" % param)))
    # print(model)
    return model

def train():
    path = "/media/ssd1T/antispoof/2019/LA"
    train_data = TorchDataset(data_list=path+"/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", data_dir="/media/ssd1T/anzhe/GRCNN/", task="train", num_classes=20)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=True)

    net = GRCNNs(2)
    if args.cuda:
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=3e-4)
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
            batch_flacs_MGD = torch.from_numpy(batch_flacs_MGD).float()
            batch_flacs_STFT = torch.from_numpy(batch_flacs_STFT).float()
            if args.cuda:
                batch_flacs_MGD = batch_flacs_MGD.cuda()
                batch_flacs_STFT = batch_flacs_STFT.cuda()

            preditcted_labels = net(batch_flacs_MGD, batch_flacs_STFT, args.cuda)

            optimizer.zero_grad()
            # print(preditcted_labels.shape)
            # print(batch_labels.shape)

            if args.cuda:
                loss = criterion(preditcted_labels, torch.from_numpy(batch_labels).cuda())
            else:
                loss = criterion(preditcted_labels, torch.from_numpy(batch_labels))
            del preditcted_labels
            loss.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), max_norm=1)
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
        if running_loss / (train_data.get_len()//args.batch_size) >= last_loss:
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

    if args.task == "dev":
        net = load_model()
        if args.cuda:
            net.cuda()
        # net.eval()
        dev(net)

    if args.task == "eval":
        net = load_model()
        if args.cuda:
            net.cuda()
        # net.eval()
        eval(net)
    
    if args.task == "dev_and_eval":
        dev_and_eval()