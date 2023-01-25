import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import scipy.io as sio
import torch.optim as optim


N_classes=5
freqs=[0.0,3.5]
data_directory='Physionet_mat/'

model_dir='Models/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
        self.fi=nn.Parameter(torch.randn(self.n_filt,1).type('torch.FloatTensor'))
        self.filt=[]

    def forward(self, x):
        u=self.u.expand(self.n_filt,self.n_time)
        fi=self.fi.expand(self.n_filt,self.n_time)
        w=self.w.expand(self.n_filt,self.n_time)*3
        s=self.s.expand(self.n_filt,self.n_time)*5
        time=self.time.expand_as(s)
        filt=torch.exp(-3.1314*torch.abs(s)*((time-u)**2))*torch.cos(2*3.1415*w*10*time+fi)
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
    

def making_batch(names,labels,prob,n_batch=12):
    data=[]
    label=[]
    
    temp=np.arange(0,len(labels))
    selected=np.random.choice(temp, size=n_batch, p=prob)
    
    for i in selected:
        
        f=sio.loadmat(data_directory+names[i].replace(' ',''))
        label.append(labels[i])
        data.append(f['SIG'])
         
    return data,label

def files_for_test(test_subjects):
    test_files=[]
    test_labels=[]
    for s in test_subjects:
        temp=sio.loadmat(data_directory+s)
        test_files=test_files+list(temp['names'])
        test_labels=test_labels+list(temp['scores'][0])
    return test_files,test_labels

#%% initial
# train_files,train_labels,val_files,val_labels,test_subjects=file_index()
# torch.save({'train_files':train_files,'train_labels':train_labels,'val_files':val_files,'val_labels':val_labels,'test_subjects':test_subjects},'Names_labels.data')

temp=torch.load('Names_labels.data')
train_files=temp['train_files']
train_labels=temp['train_labels']
val_files=temp['val_files']
val_labels=temp['val_labels']
test_subjects=temp['test_subjects']
train_files=train_files
train_labels=train_labels

train_labels, train_prob=set_labels(train_labels)
val_labels, val_prob=set_labels(val_labels)

minibach_size=16
minibach_size_val=32
running_loss = 0.0
count=0
validation_iter=128
n_iter=500000000
saving_iter=1000
lr_sch_iter=5000

LR=0.001/minibach_size

net=Net()
net=net.to(device)

criterion = nn.CrossEntropyLoss()


my_list = ['wav1.s','wav1.fi','wav1.u','wav1.w','wav2.s','wav2.fi','wav2.u','wav2.w']
params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, net.named_parameters()))))
base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, net.named_parameters()))))

optimizer = optim.Adam([{'params': base_params,'lr': (LR)},{'params': params, 'lr': (LR*50)}])

torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}
            ,model_dir+'model0')

#%% train net
def cm__(target,lebels,n=5):
    cm=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            cm[i][j]=sum((target==i)&(lebels==j))
    return cm
train_loss_=[]
val_loss_=[]


for i in range(n_iter):
    data,label=making_batch(names=train_files,labels=train_labels,prob=train_prob,n_batch=minibach_size)
    label=np.array(label)
    label= torch.tensor(label).type('torch.LongTensor')
    data=np.array(data)
    data=torch.tensor(data).type('torch.FloatTensor')
    data, label = data.to(device), label.to(device)
    optimizer.zero_grad()
    outputs = net(data[:,0,:].unsqueeze(1),data[:,1,:].unsqueeze(1))
    loss = criterion(outputs,label)
    loss.backward()
    optimizer.step()
    net.wav1.w.data.clamp_(freqs[0],freqs[1])
    net.wav2.w.data.clamp_(freqs[0],freqs[1])
    running_loss += loss.item()
    
    if i % validation_iter == (validation_iter-1):   
        data,label=making_batch(names=val_files,labels=val_labels,prob=None,n_batch=minibach_size_val)
        label= torch.tensor(label).type('torch.LongTensor')
        data=torch.tensor(data).type('torch.FloatTensor')
        data, label = data.to(device), label.to(device)
        net.eval()
        net.wav1.w.data.clamp_(freqs[0],freqs[1])
        net.wav2.w.data.clamp_(freqs[0],freqs[1])
        outputs = net(data[:,0,:].unsqueeze(1),data[:,1,:].unsqueeze(1))
        ll=label.cpu().detach().numpy()
        oo=(torch.max(outputs, 1)[1]).cpu().detach().numpy()
        print(cm__(ll,oo))
        net.train()
        val_loss = criterion(outputs,label)
        print(str(running_loss / validation_iter)+'   '+str(val_loss.cpu().detach().numpy()))
        train_loss_.append(running_loss / validation_iter)
        val_loss_.append(val_loss.cpu().detach().numpy())
        running_loss = 0.0
        
    if i % lr_sch_iter==(lr_sch_iter-1):
        optimizer.param_groups[0]['lr']=.95*optimizer.param_groups[0]['lr']
        optimizer.param_groups[1]['lr']=.95*optimizer.param_groups[1]['lr']
        print(optimizer.param_groups[0]['lr'])
        
    if i % saving_iter == (saving_iter-1):
        torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss':val_loss_,
                    'train_loss':train_loss_}
                    ,model_dir+'model_'+str(i))
            
