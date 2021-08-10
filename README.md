# GRCNN


This is a implymentation of GRCNN model of ```
@article{gomez2019gated,
  title={A gated recurrent convolutional neural network for robust spoofing detection},
  author={Gomez-Alanis, Alejandro and Peinado, Antonio M and Gonzalez, Jose A and Gomez, Angel M},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={27},
  number={12},
  pages={1985--1999},
  year={2019},
  publisher={IEEE}
}``` in pytorch.

### Status
Currently, I have finished the GRCNNs model without noise(aka, SNM CNN model). And I only finished the train part.

### How to use
#### Features
Features are put in `MGDs` and `STFTs` folders which are in the same `dir`.   
MGD is extracted with the help of `Octave`(a lightweight matlab which has nearly all matlab functions and can be used without GUI) and the [covarep](https://github.com/covarep/covarep).   
STFT is extracted with the help of `librosa` of python3.   
I have made a script called `database.py` to handle extractinng those two features mentioned above.

#### Train
`main.py` is to handle this.

#### Sketches
I have done many test in `Test.inpy`, it is messy, but if you have time to check it, you will get better understanding of some points in this model.
