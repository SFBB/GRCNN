import pandas as pd
import subprocess
import os
import torchaudio
import numpy as np
import librosa


# train_list = pd.read_csv(path, header=None, delim_whitespace=True)
# # print(train_list.at[0,])
# for row in train_list.iterrows():
#     print(row[1][1])
#     break
    
    
class Database:
    def __init__(self, index_path, data_path, list=True):
        self.index_path = index_path
        self.data_path = data_path
        
        if list:
            self.index = self.generate_index(pd.read_csv(self.index_path, header=None, delim_whitespace=True))
        else:
            self.index = self.generate_index_dir()
        
    def generate_index(self, index_):
        index = []
        for row in index_.iterrows():
            index.append({"index": row[0], "id": row[1][0], "name": row[1][1], "label": {"type": row[1][3], "gender": row[1][4]}})
        return index
        
    def generate_index_dir(self):
        index = []
        flacs = os.listdir(self.data_path)
        for flac in flacs:
            if ".flac" in flac:
                index.append({"index": "", "id": "", "name": flac.replace(".flac", "")})
        assert len(flacs) == len(index)
        return index

    def generate_MGD(self, dest, matlab_dir="matlab"):
        os.chdir(matlab_dir)    
        # for flac in self.index:
#             print(os.getcwd())
        bashCommand = """octave --eval 'run("{}", "{}")'""".format(self.data_path, dest)
        print(bashCommand)
        process = os.system(bashCommand)
#             output, error = process.communicate()
#             print(process)
        os.chdir("..")
        mats = os.listdir(dest)
        for mat in mats:
            if ".mat" in mat:
                grp_phase = np.loadtxt(dest+"/"+mat)
    #             print(grp_phase)
                with open(dest+"/"+mat.replace(".mat", ".npy"), 'wb') as f:
                    np.save(f, grp_phase)
                os.remove(dest+"/"+mat)
#             break
        
    def generate_STFT(self, dest):
        # print(waveform.numpy())
#         print(len(waveform[0]))
        n_fft = 1024
       
        window = "blackman"

        n_mels = 256
        fmin = 20
        fmax = 8000
        
        for flac in self.index:
            waveform, sample_rate = torchaudio.load(self.data_path+"/"+flac["name"]+".flac")
#             print(waveform)
            win_length = int(np.ceil(0.025*sample_rate))
            hop_length = int(np.ceil(0.010*sample_rate))
            X = librosa.stft(waveform.numpy()[0], n_fft, hop_length, win_length, window, center=True)
        #     print(np.abs(X).shape)
            frames = np.log(librosa.feature.melspectrogram(y=waveform.numpy()[0], sr=sample_rate, S=X, n_mels=n_mels, fmin=fmin, fmax=fmax) + 1e-6)
            with open(dest+"/"+flac["name"]+'.npy', 'wb') as f:
                np.save(f, np.abs(frames))

    # def load_







if __name__ == "__main__":
    path = "/media/ssd1T/antispoof/2019/LA"
    dest_path = "/media/ssd1T/anzhe/GRCNN"

    # db = Database(path+"/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", path+"/ASVspoof2019_LA_train/flac", False)
    # db.generate_MGD(dest_path+"/MGDs/train", "/home/anzhe/Documents/test/GRCNN/matlab")
    # db.generate_STFT(dest_path+"/STFTs/train")

    db = Database(path+"/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", path+"/ASVspoof2019_LA_dev/flac", False)
    db.generate_MGD(dest_path+"/MGDs/dev", "/home/anzhe/Documents/test/GRCNN/matlab")
    db.generate_STFT(dest_path+"/STFTs/dev")

    db = Database(path+"/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", path+"/ASVspoof2019_LA_eval/flac", False)
    db.generate_MGD(dest_path+"/MGDs/eval", "/home/anzhe/Documents/test/GRCNN/matlab")
    db.generate_STFT(dest_path+"/STFTs/eval")


    ### cm train
    # db = Database(path+"/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", path+"/ASVspoof2019_LA_train/flac")
    # db.generate_MGD(dest_path+"/MGDs/cm.train", "/home/anzhe/Documents/test/GRCNN/matlab")
    # db.generate_STFT(dest_path+"/STFTs/cm.train")

    # ### cm dev
    # db = Database(path+"/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trn.txt", path+"/ASVspoof2019_LA_dev/flac")
    # db.generate_MGD(dest_path+"/MGDs/cm.dev", "/home/anzhe/Documents/test/GRCNN/matlab")
    # db.generate_STFT(dest_path+"/STFTs/cm.dev")

    # ### cm eval
    # db = Database(path+"/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trn.txt", path+"/ASVspoof2019_LA_eval/flac")
    # db.generate_MGD(dest_path+"/MGDs/cm.eval", "/home/anzhe/Documents/test/GRCNN/matlab")
    # db.generate_STFT(dest_path+"/STFTs/cm.eval")

    ### asv dev
    # db = Database(path+"/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trn.txt", path+"/ASVspoof2019_LA_eval/flac")
    # db.generate_MGD(dest_path+"/MGDs/cm.eval", "/home/anzhe/Documents/test/GRCNN/matlab")
    # db.generate_STFT(dest_path+"/STFTs/cm.eval")

    ### asv eval
    