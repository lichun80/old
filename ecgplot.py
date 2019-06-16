#!/home/desktop/anaconda3/bin/python3.6
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch.autograd import Variable
from biosppy.signals import ecg

#############################################

## Load img&label array   
with open('/home/desktop/thesis/ls_ilabel.pkl','rb') as file:
        label_f = pickle.load(file)
np_label = np.array(label_f)

np_img = np.load('/home/desktop/thesis/patient_lead.npy')


x = np_img
y = np_label
a0 = x[0]
a1 = x[1]
b0 = np.reshape(a0,(12,5000))
b1 = np.reshape(a1,(12,5000))

#AF
plt.figure(figsize=(20,10))
plt.suptitle('0.AF', fontsize=16)
plt.subplot(12,1,1)
plt.plot(b0[0])
plt.subplot(12,1,2)
plt.plot(b0[1])
plt.subplot(12,1,3)
plt.plot(b0[2])
plt.subplot(12,1,4)
plt.plot(b0[3])
plt.subplot(12,1,5)
plt.plot(b0[4])
plt.subplot(12,1,6)
plt.plot(b0[5])
plt.subplot(12,1,7)
plt.plot(b0[6])
plt.subplot(12,1,8)
plt.plot(b0[7])
plt.subplot(12,1,9)
plt.plot(b0[8])
plt.subplot(12,1,10)
plt.plot(b0[9])
plt.subplot(12,1,11)
plt.plot(b0[10])
plt.subplot(12,1,12)
plt.plot(b0[11])
plt.show()

#AFL
plt.figure(figsize=(20,10))
plt.suptitle('1.AFL', fontsize=16)
plt.subplot(12,1,1)
plt.plot(b1[0])
plt.subplot(12,1,2)
plt.plot(b1[1])
plt.subplot(12,1,3)
plt.plot(b1[2])
plt.subplot(12,1,4)
plt.plot(b1[3])
plt.subplot(12,1,5)
plt.plot(b1[4])
plt.subplot(12,1,6)
plt.plot(b1[5])
plt.subplot(12,1,7)
plt.plot(b1[6])
plt.subplot(12,1,8)
plt.plot(b1[7])
plt.subplot(12,1,9)
plt.plot(b1[8])
plt.subplot(12,1,10)
plt.plot(b1[9])
plt.subplot(12,1,11)
plt.plot(b1[10])
plt.subplot(12,1,12)
plt.plot(b1[11])
plt.show()


