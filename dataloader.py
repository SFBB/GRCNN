import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
# from utils import image_processing
import os
import pandas as pd




# 以数据类型为单位，train是一个dataset，dev是一个，test是一个
class TorchDataset(Dataset):
    def __init__(self, data_list, data_dir, task, num_classes, repeat=1):
        '''
        :param data_list: 数据文件TXT：格式：imge_name.jpg label1_id labe2_id
        :param data_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :num_classes: 最后要分的类别数量，geniue+K种伪造方式
        # :param resize_height 为None时，不进行缩放
        # :param resize_width  为None时，不进行缩放，
        #                       PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        '''
        self.data_list = self.load_list(data_list)
        self.data_dir = data_dir
        self.len = len(self.data_list)
        self.task = task
        self.num_classes = num_classes
        self.repeat = repeat

        # 默认特征是STFT，可选MGD
        self.feature = "STFT"

        # padding或者剪切的固定长度为N frames
        self.fixed_length = 256
 
        # 相关预处理的初始化
        '''class torchvision.transforms.ToTensor'''
        # 把shape=(H,W,C)的像素值范围为[0, 255]的PIL.Image或者numpy.ndarray数据
        # 转换成shape=(C,H,W)的像素数据，并且被归一化到[0.0, 1.0]的torch.FloatTensor类型。
        # self.toTensor = transforms.ToTensor()
 
        '''class torchvision.transforms.Normalize(mean, std)
        此转换类作用于torch. * Tensor,给定均值(R, G, B) 和标准差(R, G, B)，
        用公式channel = (channel - mean) / std进行规范化。
        '''
        # self.normalize=transforms.Normalize()
 
    def __getitem__(self, i):
        index = i % self.len
        # print("i={},index={}".format(i, index))
        flac_name, label = self.data_list[index]["name"], self.data_list[index]["label"]
        # flac_path = os.path.join(self.data_dir, flac_name)
        if self.task != "compute":
            flac_feature_STFT = self.load_data(flac_name, "STFT")
            flac_feature_MGD = self.load_data(flac_name, "MGD")
            flac_feature_STFT = self.data_preproccess(flac_feature_STFT)
            flac_feature_MGD = self.data_preproccess(flac_feature_MGD)
        # label=np.array(label)
        label = self.process_label(label)
        if self.task == "train":
            return flac_feature_MGD, flac_feature_STFT, label
        elif self.task == "dev" or self.task == "eval":
            return flac_feature_MGD, flac_feature_STFT, self.data_list[index]["id"], flac_name, self.data_list[index]["label"]["type"], self.data_list[index]["label"]["gender"], label
        else:
            return label, flac_name, self.data_list[index]["id"]
 
    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.data_list) * self.repeat
        return data_len
 
    def load_list(self, data_list):
        index = pd.read_csv(data_list, header=None, delim_whitespace=True)
        data_list = []
        for row in index.iterrows():
            data_list.append({"index": row[0], "id": row[1][0], "name": row[1][1], "label": {"type": row[1][3], "gender": row[1][4]}})
        return data_list

    def load_data(self, file_name, feature):
        '''
        加载数据
        :param path:
        :param resize_height:
        :param resize_width:
        :param normalization: 是否归一化
        :return:
        '''
        with open(self.data_dir+"/"+feature+"s/"+self.task+"/"+file_name+".npy", "rb") as file:
            data = np.load(file)
        if feature == "MGD":
            data = data[:256, :]
        elif feature == "STFT":
            data = STFT = (data - np.mean(data)) / np.std(data)
        # image = image_processing.read_image(path, resize_height, resize_width, normalization)
        return data

    def process_label(self, label):
        # "label": {"type": row[1][3], "gender": row[1][4]}}
        # label_ = np.zeros(self.num_classes)
        if self.task != "compute":
            if label["gender"] == "bonafide": # geniue    - bonafide
                label_ = 0
            else: # spoof - check type    A03 spoof
                # label_ = int(label["type"][1:])
                label_ = 1
            return label_
        else:
            if label["gender"] == "bonafide": # geniue    - bonafide
                label_ = 0
            else: # spoof - check type    A03 spoof
                label_ = int(label["type"][1:])
                # label_ = 1
            return label_

    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        if data.shape[1] > self.fixed_length:
            data = data[:, (data.shape[1]-self.fixed_length)//2:(data.shape[1]-self.fixed_length)//2+self.fixed_length]
        else:
            data = np.pad(data, ((0, 0), (0, self.fixed_length-data.shape[1])), mode="wrap")
        data = data.reshape(1, 256, self.fixed_length) # (1, 1, 256, frames_num), (N, C, W, H)
        # data = torch.from_numpy(data)
        return data

    def select_feature(self, feature_name):
        self.feature = feature_name

    def get_len(self):
        return self.len



# ————————————————
# 版权声明：本文为CSDN博主「pan_jinquan」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/guyuealian/article/details/88343924



# 自定义的函数用于DataLoader的batch collect，以解决使用的数据长度不一的问题
def custom_collate(batch):
    # index = 0
    batch_flacs_MGD = []
    batch_flacs_STFT = []
    batch_labels = []
    for flac in batch:
        batch_flacs_MGD.append(flac[0])
        batch_flacs_STFT.append(flac[1])
        batch_labels.append(flac[2])
        # print(flac(1))
        # batch_labels = torch.cat((batch_labels, torch.from_numpy(flac[1])))
        # index += 1
    # batch_flacs: (32, 1, 256, frame_number)
    # batch_labels: (32, num_classes)
    # print(len(batch))
    # print(len(batch[0]))
    # print(len(batch[0][0]))
    # print(len(batch[0][0][0]))
    # print(len(batch[0][0][0][0]))
    # print(batch)
    return np.array(batch_flacs_MGD), np.array(batch_flacs_STFT), np.array(batch_labels)


def custom_collate_for_dev(batch):
    # index = 0
    batch_flacs_MGD = []
    batch_flacs_STFT = []
    batch_ids = []
    batch_names = []
    batch_types = []
    batch_genders = []
    batch_labels = []
    for flac in batch:
        batch_flacs_MGD.append(flac[0])
        batch_flacs_STFT.append(flac[1])
        batch_ids.append(flac[2])
        batch_names.append(flac[3])
        # batch_ids.append(flac[4])
        batch_types.append(flac[4])
        batch_genders.append(flac[5])
        batch_labels.append(flac[6])
        # print(flac(1))
        # batch_labels = torch.cat((batch_labels, torch.from_numpy(flac[1])))
        # index += 1
    # batch_flacs: (32, 1, 256, frame_number)
    # batch_labels: (32, num_classes)
    # print(len(batch))
    # print(len(batch[0]))
    # print(len(batch[0][0]))
    # print(len(batch[0][0][0]))
    # print(len(batch[0][0][0][0]))
    # print(batch)
    return np.array(batch_flacs_MGD), np.array(batch_flacs_STFT), batch_ids, batch_names, batch_types, batch_genders, np.array(batch_labels)


def custom_collate_for_compute(batch):
    # index = 0
    # batch_flacs_MGD = []
    # batch_flacs_STFT = []
    batch_labels = []
    batch_names = []
    batch_ids = []
    for flac in batch:
        # batch_flacs_MGD.append(flac[0])
        # batch_flacs_STFT.append(flac[1])
        batch_labels.append(flac[0])
        batch_names.append(flac[1])
        batch_ids.append(flac[2])
        # print(flac(1))
        # batch_labels = torch.cat((batch_labels, torch.from_numpy(flac[1])))
        # index += 1
    # batch_flacs: (32, 1, 256, frame_number)
    # batch_labels: (32, num_classes)
    # print(len(batch))
    # print(len(batch[0]))
    # print(len(batch[0][0]))
    # print(len(batch[0][0][0]))
    # print(len(batch[0][0][0][0]))
    # print(batch)
    return np.array(batch_labels), batch_names, batch_ids

if __name__ == "__main__":

    path = "/media/ssd1T/antispoof/2019/LA"
    train_data = TorchDataset(data_list=path+"/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", data_dir="./", num_classes=20, repeat=None)
    train_loader = DataLoader(dataset=train_data, batch_size=32, collate_fn=custom_collate, shuffle=True)

    for step, (batch_flacs, batch_labels) in enumerate(train_loader):
        # print(step)
        # print(batch_flacs, batch_labels)
        print(len(batch_flacs), len(batch_labels))
        print(batch_flacs[0].shape)
        print(batch_labels[0].shape)
        pass