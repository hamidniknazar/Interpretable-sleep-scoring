import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import scipy.io as sio
import sklearn.metrics as metric
os.environ['KMP_DUPLICATE_LIB_OK']='True'

N_classes=5

data_directory='Physionet_mat/'

model_dir='Models/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

        x1 =F.relu(self.wav1(x1))
        x2 =F.relu(self.wav2(x2))
        x1=torch.cat((x1,x2),1)
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
        return self.wav1.return_filt()
    
    
def one_hot(label,batch_size,n_out):
    oo=np.zeros([batch_size,n_out])
    for i in range(batch_size):
        oo[i,label[i]]=1
    return oo

#%% data manager



def file_index(n_test=15,n_val=3):
    files=os.listdir(data_directory)
    mat_files=[files[i] for i in range(len(files)) if '.mat' in files[i]]
    
    indeces=np.arange(0,len(mat_files))
    np.random.shuffle(indeces)
    test_subjects=[mat_files[i] for i in indeces[0:n_test]]
    val_subjects=[mat_files[i] for i in indeces[n_test:n_test+n_val]]
    train_subjects=[mat_files[i] for i in indeces[n_test+n_val:]]
    
    
    train_files=[]
    train_labels=[]
    for s in train_subjects:
        temp=sio.loadmat(data_directory+s)
        train_files=train_files+list(temp['names'])
        train_labels=train_labels+list(temp['scores'][0])
           
    val_files=[]
    val_labels=[]
    for s in val_subjects:
        temp=sio.loadmat(data_directory+s)
        val_files=val_files+list(temp['names'])
        val_labels=val_labels+list(temp['scores'][0])
           
    
    return train_files,train_labels,val_files,val_labels,test_subjects

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
    labels_=set(labels)
    probabilities=np.ones_like(labels,dtype=np.float64)
    for c in range(len(labels_)):
        count=np.sum(labels==c)
        probabilities[labels==c]=1/count
    probabilities=probabilities/sum(probabilities)
    return labels, probabilities
    

def making_batch(names,labels,prob,n_batch=12,ite=1):
    data=[]
    label=[]

    selected=np.arange(n_batch*(ite-1),n_batch*ite)
    
    for i in selected:
        
        f=sio.loadmat(data_directory+names[i].replace(' ',''))
        label.append(labels[i])
        data.append(f['SIG'])
         
    return data,label

def cm__(target,lebels,n=5):
    cm=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            cm[i][j]=sum((target==i)&(lebels==j))
    return cm
train_loss_=[]
val_loss_=[]

#%% initial

temp=torch.load('Names_labels.data')
test_subjects=temp['test_subjects']
test_subjects=[name.replace('.mat','') for name in test_subjects ]

test_files,test_labels=files_for_test(test_subjects)

test_labels,_=set_labels(test_labels)


temp=torch.load(model_dir+'model_366999', map_location=lambda storage, loc: storage)
net=Net()
net=net.to(device)
net.load_state_dict(temp['model_state_dict'])

net.eval()


test_minibach_size=20
n_iter=len(test_labels)/test_minibach_size

cm=np.zeros([5,5])
L=[]
out=[]
for i in range(int(n_iter)):
    if i%50==49:
        print(str(i+1)+'/'+str(int(n_iter)))
        kappa_LSTM=metric.cohen_kappa_score(out,L)
        print(cm)
        print(kappa_LSTM)
    data,label=making_batch(names=test_files,labels=test_labels,prob=None,n_batch=test_minibach_size,ite=i)
    L=L+label
    label=np.array(label)
    label= torch.tensor(label).type('torch.LongTensor')
    data=np.array(data)
    data=torch.tensor(data).type('torch.FloatTensor')
    
    data, label = data.to(device), label.to(device)
    outputs = net(data[:,0,:].unsqueeze(1),data[:,1,:].unsqueeze(1))
    out=out+list(torch.argmax(outputs,1).cpu().numpy())
    cm+=cm__(label.cpu().numpy(),torch.argmax(outputs,1).cpu().numpy())

print((cm).astype(int))
acc=np.sum(np.diag(cm))/np.sum(cm)
print(acc)


kappa_LSTM=metric.cohen_kappa_score(out,L)
print(kappa_LSTM)





