import os
import numpy as np
from random import shuffle
import torch.nn as nn
import torch.nn.functional as F
import torch
import scipy.io as sio
import torch.optim as optim
import random
from torch import autograd
import matplotlib.pyplot as plt


N_classes=5
N_wave=8
num_layers=2
N_around=4

data_directory='Physionet_mat/'

model_dir='Models/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%% LSTM net

class Net_LSTM(nn.Module):
    def __init__(self,N_eeg_wave=32,N_eog_wave=8,wave_time=2,fs=100):
        super(Net_LSTM, self).__init__()
        self.lstm_forward=nn.LSTM(N_classes,N_classes*2,num_layers=num_layers)
        self.lstm_backward=nn.LSTM(N_classes,N_classes*2,num_layers=num_layers)
        self.fc=nn.Linear(4*N_classes, N_classes)
        self.dr=nn.Dropout(p=0.5)
    def forward(self, x_for,x_back):
        _, (x_for,_)=self.lstm_forward(x_for)
        _, (x_back,_)=self.lstm_backward(x_back)
        x_for=x_for[-1,:]
        x_back=x_back[-1,:]
        x=torch.cat([x_for,x_back],1)
        x=self.dr(x)   
        x=self.fc(x)
        return x

#%% define wave layer
class Wave(nn.Module):
    def __init__(self,n_filt,n_time,n_in=1,strid=1):
        super(Wave, self).__init__()
        self.n_filt=n_filt
        self.n_time=n_time
        self.n_in=n_in
        self.strid=strid
        self.time=((torch.unsqueeze(torch.tensor(range(self.n_time)),1).t().type('torch.FloatTensor')+1-n_time/2)/100).to(device)
        self.u=nn.Parameter(torch.randn(self.n_filt,1).type('torch.FloatTensor'))
        self.w=nn.Parameter(torch.randn(self.n_filt,1).type('torch.FloatTensor'))
        self.s=nn.Parameter(torch.randn(self.n_filt,1).type('torch.FloatTensor'))
        self.filt=[]

    def forward(self, x):
        u=self.u.expand(self.n_filt,self.n_time)
        w=self.w.expand(self.n_filt,self.n_time)*3
        s=self.s.expand(self.n_filt,self.n_time)*5
        time=self.time.expand_as(s)
        filt=torch.exp(-3.1314*torch.abs(s)*((time-u)**2))*torch.cos(2*3.1415*w*10*time)
        self.filt=filt.to(device)
        filt=torch.unsqueeze(filt,1) 
        filt=filt.repeat(1,self.n_in,1)
        return F.conv1d(x,filt,stride=self.strid)
    def return_filt(self):
        return self.filt
        


#%% network definition
class Net(nn.Module):
    def __init__(self,N_eeg_wave=32,N_eog_wave=8,wave_time=2,fs=100):
        super(Net, self).__init__()
        self.wav1=Wave(N_eeg_wave,wave_time*fs)
        self.wav2=Wave(N_eog_wave,wave_time*fs)
        # self.wav1=self.wav1.to(device)
        self.pool = nn.MaxPool1d(3)
        self.mixing = nn.Linear(N_eeg_wave+N_eog_wave,256)
        self.conv1 = nn.Conv1d(256, 64, 3)
        self.conv1_bn = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 3)
        self.conv2_bn = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, 3)
        self.conv3_bn = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, 3)
        self.conv4_bn = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 256, 3)
        self.conv5_bn = nn.BatchNorm1d(256)
        

        self.fc1 = nn.Linear(2560, 256)
        self.fc2 = nn.Linear(256, 100)
        self.fc3 = nn.Linear(100, N_classes)
        self.dr2=nn.Dropout(p=0.5)

    def forward(self, x1,x2):
        x1=F.normalize(x1)
        x2=F.normalize(x2)

        self.O_W1 =F.relu(self.wav1(x1))
        self.O_W1.retain_grad()
        
        self.O_W2 =F.relu(self.wav2(x2))
        self.O_W2.retain_grad()
        x1=torch.cat((self.O_W1,self.O_W2),1)
        x1=torch.transpose(x1, 1, 2) 
        
        x1=self.mixing(x1)
        x1=torch.transpose(x1, 2, 1) 

        x1 = self.pool(F.relu(self.conv1(x1)))
        x1 = self.conv1_bn(x1)
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = self.conv2_bn(x1)
        x1 = self.pool(F.relu(self.conv3(x1)))
        x1 = self.conv3_bn(x1)
        x1 = self.pool(F.relu(self.conv4(x1)))
        x1 = self.conv4_bn(x1)
        x1 = self.pool(F.relu(self.conv5(x1)))
        x1 = self.conv5_bn(x1)
   
        x = x1.view(-1, self.num_flat_features(x1))
        x=self.dr2(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
#        print('jhbkj'+str(num_features))
        return num_features
    def plot_wave(self):
        return self.wav1.return_filt(),self.wav2.return_filt()
    
#%% batch
def making_batch(names,labels,prob,n_batch=12,ite=1):
    data=[]
    label=[]

    selected=np.arange(n_batch*(ite-1),n_batch*ite)
    
    for i in selected:
        
        f=sio.loadmat(data_directory+names[i].replace(' ',''))
        label.append(labels[i])
        data.append(f['SIG'])
         
    return data,label


#%% loading

def files_for_test(test_subjects):
    test_files=[]
    test_labels=[]
    for s in test_subjects:
        temp=sio.loadmat(data_directory+s)
        test_files=test_files+list(temp['names'])
        test_labels=test_labels+list(temp['scores'][0])
    return test_files,test_labels
    


def set_labels(labels):
    
    labels=np.array(labels)
    labels[labels==4]=3
    labels[labels==5]=4
    
    return labels


temp=torch.load('Names_labels.data')
# train_files=temp['train_files']
# train_labels=temp['train_labels']
test_subjects=temp['test_subjects']

test_files,test_labels=files_for_test(test_subjects)

# train_labels=set_labels(train_labels)

test_labels=set_labels(test_labels)
temp=torch.load(model_dir+'model_366999', map_location=lambda storage, loc: storage)
net=Net()
net=net.to(device)
net.load_state_dict(temp['model_state_dict'])
net.eval()
criterion = nn.CrossEntropyLoss()

minibach_size=1
data,label=making_batch(names=test_files,labels=test_labels,prob=None,n_batch=minibach_size,ite=120)
label= torch.tensor(label).type('torch.LongTensor')
data=torch.tensor(data).type('torch.FloatTensor')
data, label = data.to(device), label.to(device)
outputs = net(data[:,0,:].unsqueeze(1),data[:,1,:].unsqueeze(1))

gabor_eeg,gabor_eog=net.plot_wave()
gabor_eeg,gabor_eog=gabor_eeg.cpu().detach().numpy(),gabor_eog.cpu().detach().numpy()
time=np.arange(1,201)/100
freq=np.arange(0,100)/2

#%% load LSTM net
temp=torch.load(model_dir+'LSTM_319999', map_location=lambda storage, loc: storage)
net_LSTM=Net_LSTM()
net_LSTM=net_LSTM.to(device)
net_LSTM.load_state_dict(temp['model_state_dict'])
net_LSTM.eval()

#%% interpereting
def making_batch(names,labels,selected=1):
    data=[]
    label=[]
    label.append(labels[selected])
    for i in range(selected-N_around,selected+N_around+1):
        
        f=sio.loadmat(data_directory+names[i].replace(' ',''))
        data.append(f['SIG'])     
    return data,label

# def making_batch(F_in,L_in,n_batch,prob):
#     data_1=[]
#     data_2=[]
#     label=[]
    
    
#     temp=np.arange(N_around+1,F_in.shape[0]-N_around-1)
#     selected=np.random.choice(temp, size=n_batch, p=prob)
#     for i in selected:
#         label.append(L_in[i])
#         data_1.append(F_in[i-N_around:i+1,:])
#         data_2.append(F_in[i:i+N_around+1,:])
         
#     return data_1,data_2,label

for selected in [4500,4258 , 4570]:

    data, label=making_batch(names=test_files,labels=test_labels,selected=selected)
    data=torch.tensor(data).type('torch.FloatTensor')
    data= data.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    optimizer.zero_grad()
    net.eval()
    
    o=net(data[:,0,:].unsqueeze(1),data[:,1,:].unsqueeze(1))
    
    # loss = criterion(o[N_around,:].unsqueeze(0),torch.argmax(o[N_around,:]).unsqueeze(0))
    # loss.backward(retain_graph=True)
    
    o_wav1=net.O_W1.cpu().detach().numpy()
    o_wav2=net.O_W2.cpu().detach().numpy()
    
    
    o[N_around,torch.argmax(o[N_around,:])].backward(retain_graph=True)
    
    G=net.O_W1.grad.cpu().numpy()[N_around,:]
    wg=o_wav1[N_around,:]*G
    
    wg[wg<0]=0
    wg=abs(wg)
    
    
    bbb=np.sum((wg),1)
    
    
    aa=np.argsort(bbb)
        
    To_save={}
    plt.figure()
    
    
    for j in range(1,N_wave+1):
        plt.subplot(3,N_wave,j)
        # if j<5:
        plt.plot(time-1,gabor_eeg[aa[-j],:])
        # plt.title('W*G = ' + '%.2f' % wg[aa[-j]])
        # else:
        #     plt.plot(time,gabor_eeg[aa[j-5],:])
        #     plt.title('W*G = ' + '%.2f' % wg[aa[j-5]])
        # plt.xticks([])
        # plt.yticks([])
        plt.title('('+ str(aa[-j]) + ')')
        
    To_save['kernels']=aa
        
    plt.subplot(3,1,2)
    plt.imshow((wg[aa[len(aa)-1:len(aa)-1-N_wave:-1]])**(1/4),aspect='auto',interpolation='None',origin='lower',cmap='jet')
    
    To_save['image_impact']=wg[aa[len(aa)-1:len(aa)-1-N_wave:-1]]
    
    
    
    # plt.imshow(aaa[aa[len(aa)-1:0:-1]],aspect='auto',interpolation='None',origin='lower')
    plt.yticks(np.arange(0,N_wave))
    
    
    
    plt.subplot(3,1,3)
    plt.plot(data.cpu().detach().numpy()[N_around,0,100:-100])
    plt.xlim(0,2801) 
    plt.xlabel('Time (Sec)')
    
    To_save['EEG']=data.cpu().detach().numpy()[N_around,0,100:-100]
    
    print('real label = '+str(label[0]))
    To_save['label']=label[0]
    
    sio.savemat('inter_'+str(selected)+'.mat',To_save)
    
    
