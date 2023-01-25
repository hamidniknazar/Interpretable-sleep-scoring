import numpy as np
import mne
import scipy.signal as scisig
import os
import sys
import scipy.io as sio

FS=100
l_win=int(30*FS)

edf_dir='sleep-edf-database-expanded-1.0.0/sleep-cassette/'
save_dir='Physionet_mat/'

filt=scisig.firwin(50, [.3,35], width=None, window='hamming', pass_zero='bandpass', scale=True, nyq=None, fs=FS)

files=os.listdir(edf_dir)
EDFs=[f for f in files if "-PSG.edf" in f]

annotation_desc_2_event_id = {'Sleep stage W': 0,
                              'Sleep stage 1': 1,
                              'Sleep stage 2': 2,
                              'Sleep stage 3': 3,
                              'Sleep stage 4': 4,
                              'Sleep stage R': 5}




def DOOOO(file_name):
    names=[]
    for f in files:
        if (file_name[0:7] in f) and ('Hypnogram.edf' in f) :
            score_file=f
            break
    edf_file=edf_dir+file_name
    score_file=edf_dir+score_file
    
    score=mne.read_annotations(score_file) 
    data = mne.io.read_raw_edf(edf_file,exclude=['Resp oro-nasal','EMG submental','Temp rectal','Event marker','EEG Pz-Oz'])
    data.set_annotations(score, emit_warning=False)
    score, _ = mne.events_from_annotations(data, event_id=annotation_desc_2_event_id, chunk_duration=30.)
    X = data.get_data()
    score=score[:,2]   

       
    Data=np.zeros([2,len(X[0,:])])
    Data[0,:]=scisig.filtfilt(filt, 1, X[0,:])
    Data[1,:]=scisig.filtfilt(filt, 1, X[1,:])
    Data=[Data[:,i*l_win:(i+1)*l_win] for i in range(int(Data.shape[1]/l_win))]
    
    
    if not os.path.exists(save_dir+file_name.replace('.edf','')):
        os.mkdir(save_dir+file_name.replace('.edf',''))
    
    for win in range(min(len(Data),len(score))):
        a={'SIG':Data[win],'S':score[win]}
        sio.savemat(save_dir+file_name.replace('.edf','')+'/'+str(win)+'.mat',a,do_compression=True)
        names.append(file_name.replace('.edf','')+'/'+str(win)+'.mat')
        
    
    info={'names':names,'scores':score}   
    sio.savemat(save_dir+file_name.replace('.edf','.mat'),info)
    


for edf in EDFs:
    print(edf)
    DOOOO(edf)

