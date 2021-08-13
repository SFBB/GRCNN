import torch

from dataloader import TorchDataset, custom_collate, custom_collate_for_compute
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import bob.measure
import numpy as np
from evaluate_tDCF_asvspoof19 import evaluate_tDCF_asvspoof19



def sort_fun(x):
    if not isinstance(x, str):
        return x%20
    else:
        return 20

def compute_eer(scores_path, data, result_path):
    # scores_path is to target scroes files which are gotten to be evaluated
    # dataloader_iter is the dataloader_iter which is related to the target, it is used to get the detailed classes
    dataloader = DataLoader(dataset=data, batch_size=1, collate_fn=custom_collate_for_compute, shuffle=True)
    dataloader_iter = iter(dataloader)
    # print(scores_path)
    index = pd.read_csv(scores_path, header=None, delim_whitespace=True)
    # print(index)
    scores = {}
    # scores["LA_E_5516706"] = []
    for row in index.iterrows():
        # print(row[0], row[1][0], row[1][1], row[1][2])
        scores[row[1][0]] = [row[1][1], row[1][2]]
    t = tqdm(range(data.get_len()))
    negative_labels = {"all": []} # {"A_k": []}
    positive_labels = []
    for step in t:
        (batch_labels, batch_names, batch_ids) = next(dataloader_iter)
        if batch_names[0] in scores.keys():
            # print(scores[batch_names[0]])
            if batch_labels[0]>0:
                if batch_labels[0] not in negative_labels.keys():
                    negative_labels[batch_labels[0]] = [scores[batch_names[0]][0]]
                else:
                    negative_labels[batch_labels[0]].append(scores[batch_names[0]][0])
                negative_labels["all"].append(scores[batch_names[0]][0])
            else:
                positive_labels.append(scores[batch_names[0]][0])
            # print(batch_labels[0])
            # pass
    with open(result_path, "w+") as file:
        file.write("{:<20} {:<20} {:<20} {:<20}\n".format("Attack Type", "EER", "FPR", "FNR"))
        for A_k in sorted(negative_labels.keys(), key=sort_fun):
            eer, fpr, fnr = bob.measure.eer(negative_labels[A_k], positive_labels, also_farfrr=True)
            file.write("{:<20} {:<20.5f} {:<20.5f} {:<20.5f}\n".format(str(A_k)+":", eer, fpr, fnr))
    print("Finished!")
    # print(positive_labels[0:10])
    # print(negative_labels.keys())
    # print(negative_labels[1][0:10])
    # print(negative_labels["all"][0:10])


def score(p):
    return 5*np.log(np.exp(p)/np.exp(1-p))

def compute_tDCF_and_eer_for_dev():
    ASV_SCOREFILE = "/media/ssd1T/antispoof/2019/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"
    ASV_INDEXFILE = "/media/ssd1T/antispoof/2019/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.gi.trl.txt"
    CM_SCOREFILE = "cm.dev.scores.txt"
    cm_index = pd.read_csv(CM_SCOREFILE, header=None, delim_whitespace=True)
    cm_dict = {}
    for row in cm_index.iterrows():
        cm_dict[row[1][1]] = [row[1][0], row[1][1], row[1][2], row[1][3], row[1][4], row[1][5]]
    asv_index = pd.read_csv(ASV_INDEXFILE, header=None, delim_whitespace=True)
    
    asv_scores_index = pd.read_csv(ASV_SCOREFILE, header=None, delim_whitespace=True)
    with open("asv.dev.scores.txt.standard", "w+") as asv, open("cm.dev.scores.txt.standard", "w+") as cm:
        for row in asv_index.iterrows():
            asv.write("{} {} {} {} {} {}\n".format(row[1][0], row[1][1], "-", row[1][2], row[1][3], asv_scores_index[2][row[0]]))
            cm.write("{} {} {} {} {} {}\n".format(cm_dict[row[1][1]][0], cm_dict[row[1][1]][1], cm_dict[row[1][1]][2], cm_dict[row[1][1]][3], cm_dict[row[1][1]][4], cm_dict[row[1][1]][5]))
    # with open(CM_SCOREFILE):
    #     pass
    evaluate_tDCF_asvspoof19("cm.dev.scores.txt.standard", "asv.dev.scores.txt.standard", False)

def compute_tDCF_and_eer_for_eval():
    ASV_SCOREFILE = "/media/ssd1T/antispoof/2019/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
    ASV_INDEXFILE = "/media/ssd1T/antispoof/2019/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt"
    CM_SCOREFILE = "cm.eval.scores.txt"
    cm_index = pd.read_csv(CM_SCOREFILE, header=None, delim_whitespace=True)
    cm_dict = {}
    for row in cm_index.iterrows():
        cm_dict[row[1][1]] = [row[1][0], row[1][1], row[1][2], row[1][3], row[1][4], row[1][5]]
    asv_index = pd.read_csv(ASV_INDEXFILE, header=None, delim_whitespace=True)
    
    asv_scores_index = pd.read_csv(ASV_SCOREFILE, header=None, delim_whitespace=True)
    with open("asv.eval.scores.txt.standard", "w+") as asv, open("cm.eval.scores.txt.standard", "w+") as cm:
        for row in asv_index.iterrows():
            asv.write("{} {} {} {} {} {}\n".format(row[1][0], row[1][1], "-", row[1][2], row[1][3], asv_scores_index[2][row[0]]))
            cm.write("{} {} {} {} {} {}\n".format(cm_dict[row[1][1]][0], cm_dict[row[1][1]][1], cm_dict[row[1][1]][2], cm_dict[row[1][1]][3], cm_dict[row[1][1]][4], cm_dict[row[1][1]][5]))
    # with open(CM_SCOREFILE):
    #     pass
    evaluate_tDCF_asvspoof19("cm.eval.scores.txt.standard", "asv.eval.scores.txt.standard", False)



def eval_eval():
    path = "/media/ssd1T/antispoof/2019/LA"
    scores_path = "cm.eval.scores.txt"
    eval_data = TorchDataset(data_list=path+"/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", data_dir="/media/ssd1T/anzhe/GRCNN/", task="compute", num_classes=20, repeat=None)
    # eval_loader = DataLoader(dataset=eval_data, batch_size=1, collate_fn=custom_collate_for_compute, shuffle=True)
    compute_eer(scores_path, eval_data, "cm.eval.eer_result.txt")

def eval_dev():
    path = "/media/ssd1T/antispoof/2019/LA"
    scores_path = "cm.dev.scores.txt"
    eval_data = TorchDataset(data_list=path+"/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt", data_dir="/media/ssd1T/anzhe/GRCNN/", task="compute", num_classes=20, repeat=None)
    # dev_loader = DataLoader(dataset=dev_data, batch_size=1, collate_fn=custom_collate_for_compute, shuffle=True)
    compute_eer(scores_path, eval_data, "cm.dev.eer_result.txt")

if __name__ == "__main__":
    # eval_dev()
    # eval_eval()
    compute_tDCF_and_eer_for_dev()
    compute_tDCF_and_eer_for_eval()