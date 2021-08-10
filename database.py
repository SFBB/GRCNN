import pandas as pd
import subprocess
import os
import torchaudio
import numpy as np
import librosa


path = "/media/ssd1T/antispoof/2019/LA"
# train_list = pd.read_csv(path, header=None, delim_whitespace=True)
# # print(train_list.at[0,])
# for row in train_list.iterrows():
#     print(row[1][1])
#     break
    
    
class Database:
    def __init__(self, index_path, data_path):
        self.index_path = index_path
        self.data_path = data_path
        
        self.index = self.generate_index(pd.read_csv(self.index_path, header=None, delim_whitespace=True))
        
    def generate_index(self, index_):
        index = []
        for row in index_.iterrows():
            index.append({"index": row[0], "id": row[1][0], "name": row[1][1], "label": {"type": row[1][3], "gender": row[1][4]}})
        return index
        
    def generate_MGD(self, dest, matlab_dir="matlab"):
        os.chdir(matlab_dir)    
        for flac in self.index:
#             print(os.getcwd())
            bashCommand = """octave --eval 'run("{}.flac", "{}", "{}")'""".format(self.data_path+"/"+flac["name"], flac["name"], "../"+dest)
            # print(bashCommand)
            process = os.system(bashCommand)
#             output, error = process.communicate()
#             print(process)
            grp_phase = np.loadtxt("../"+dest+"/"+flac["name"]+".mat")
#             print(grp_phase)
            with open("../"+dest+"/"+flac["name"]+'.npy', 'wb') as f:
                np.save(f, grp_phase)
            os.remove("../"+dest+"/"+flac["name"]+'.mat')
#             break
        os.chdir("..")
        
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
    db = Database(path+"/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", path+"/ASVspoof2019_LA_train/flac")
    # db.generate_MGD("MGDs", "/home/anzhe/Documents/test/GRCNN/matlab")
    db.generate_STFT("STFTs")