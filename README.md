# GRCNN


This is a implementation of GRCNN model of
```bibtex
@article{gomez2019gated,
  title={A gated recurrent convolutional neural network for robust spoofing detection},
  author={Gomez-Alanis, Alejandro and Peinado, Antonio M and Gonzalez, Jose A and Gomez, Angel M},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={27},
  number={12},
  pages={1985--1999},
  year={2019},
  publisher={IEEE}
}
```
in pytorch.

### Status
Currently, I have finished the GRCNNs model without noise(aka, SNM CNN model). And I only finished the train part.

### How to use
#### Features
Features are put in `MGDs` and `STFTs` folders which are in the same `dir`.   
MGD is extracted with the help of `Octave`(a lightweight matlab which has nearly all matlab functions and can be used without GUI) and the [covarep](https://github.com/covarep/covarep).   
STFT is extracted with the help of `librosa` of python3.   
I have made a script called `main_database.py` to handle extractinng those two features mentioned above.

#### Train
`main.py` is to handle this.   
Based on the pratical situation, you need to change the fixed_length in dataset in `dataloader.py`, to utlize your GPU memory or memory more efficiently.

#### Evalution
`main.py` --task dev/eval/dev_and_eval is used to calculate the accuracy on dev and eval, and make a general scores files which are used as index for [ASVSpoof metrics, t-DCF and EER](https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf), APPENDIX.   
`main_plus_asv.py` is used after the above step, to calculate EER and t-DCF based on the official python methods provided by ASVSpoof. (To be exact, this is only a script which can make you use official python methods more easily). 

#### Sketches
I have done many test in `Test.inpy`, it is messy, but if you have time to check it, you will get better understanding of some points in this model.

#### History
2021.08.10 Complte train function.   
2021.08.13 Fixed bugs of the model(traning loss no imporoved, very slow training speed, etc), and add eval functions. More fixes on generate features scripts and others. Imporvments on dataloader method.   

Future plan: Add classifier like GMM, PLDA to replace FC+Softmax during training phase which the original paper mentioned, to make this model work in pratice.
