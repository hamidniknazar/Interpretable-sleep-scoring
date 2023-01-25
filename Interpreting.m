clc
clear
close all

load ('Inter_Info_extended.mat')
load ('Gabor_EEG_parameters.mat')

stage_names={'Wake', 'S1','S2','SWS','REM'};

net_label=net_label(5:41124);
Power_eeg_pos=Power_eeg_pos(5:41124,:);
Power_eeg_neg=Power_eeg_neg(5:41124,:);
Power_eog_pos=Power_eog_pos(5:41124,:);
real_label=real_label(5:41124);

EEG_power=EEG_power(5:41124);
EOG_power=EOG_power(5:41124);

power_ratio=EEG_power./EOG_power;


figure
tiledlayout(2,4);
for i=[ 2 3 4 12 18  23 24 25]
    nexttile
    plot((1:200)/100-1,gabor_eeg(i,:))
    
    if (i==2 || i==18)
        ylabel('Amplitude')
    end
    xlabel('Time (Sec)')
    ti={['(' num2str(i) ')']; ['f=' num2str(w(i),'%0.1f') '  u=' num2str(u(i),'%0.2f') '  s=' num2str(s(i),'%0.2f')]};
    title(ti)
    
end



figure
tiledlayout(2,4);
for i=[ 2 3 4 12 18  23 24 25]
    nexttile
    ff=abs(fft(gabor_eeg(i,:)));
    ff=ff(1:101);
    plot(0:.5:50,ff)
    if (i==2 || i==18)
        ylabel('|FFT|')
    end
    xlabel('Freq (Hz)')
    
    
end



figure
for i=1:32
    subplot(4,8,i)
    plot((1:200)/100,gabor_eeg(i,:))
    
    if (mod(i,8)==1)
        ylabel('Amplitude')
    end
    if i>24
        xlabel('Time (Sec)')
    end
    ti={['(' num2str(i) ')']; ['f=' num2str(w(i),'%0.1f') '  u=' num2str(u(i),'%0.2f') '  s=' num2str(s(i),'%0.2f')]};
    title(ti)
end


figure
for i=1:32
    subplot(4,8,i)
    ff=abs(fft(gabor_eeg(i,:)));
    ff=ff(1:101);
    plot(0:.5:50,ff)
    if (mod(i,8)==1)
        ylabel('|fft|')
    end
    if i>24
        xlabel('Freq (Hz)')
    end
    ti={['(' num2str(i) ')']; ['f=' num2str(w(i),'%0.1f') '  u=' num2str(u(i),'%0.2f') '  s=' num2str(s(i),'%0.2f')]};
    title(ti)
end




N=1;
figure

for i = 0:4
    subplot(1,5,i+1)
    temp=Power_eeg_pos(net_label==i,:);
    [B I] = sort(temp,2,'descend');
    I=I(:,1:N);
    I=reshape(I,size(I,1)*size(I,2),1);
    Gabor_label={};
    Gabor_count=[];
    count=1;
    for j=1:32
        %         if sum(I==j)/length(I) > (1/365)
        Gabor_count(j)=sum(I==j);
        %         count=count+1;
        %         end
    end
    
    
    labels={};
    for j =1:32
        if Gabor_count(j)>sum(Gabor_count)/180
            labels{j}=[num2str(j)];
        else
            labels{j}='';
        end
    end
    pie(Gabor_count,labels)
    title([stage_names{i+1}])
    
    sgtitle(['N=' num2str(N)])
    
end




figure
N=8;
importance_i=[];
importance_v=[];

for stage=0:4
    Pow=Power_eeg_pos(real_label==stage,:);
    Pow=mean(Pow,1);
    
    [a,b]=sort(Pow);
    importance_i(stage+1,:)=b(end:-1:1);
    importance_v(stage+1,:)=a(end:-1:1);
    
    
    
    subplot(1,5,stage+1)
    temp=zeros(32,1);
    temp(importance_i(stage+1,1:N))=importance_v(stage+1,1:N);
    label={};
    for i =1:32
        label{i}=' ';
    end
    for i =1:N
        label{importance_i(stage+1,i)}=num2str(importance_i(stage+1,i));
    end
    pie(temp,label)
    title([stage_names{stage+1}])
    
    %     plot(b(end:-1:1),'-.')
    %     hold all
end
%
%
% Power_eeg_temp={};
% Power_eeg_temp_std={};
% for stage=0:4
%     temp=[];
%     temp2=[];
%     for i=1:32
%         temp=Power_eeg_pos(real_label==stage,i);
%         TF = ~isoutlier(temp,'percentiles',[10 90]);
%         temp=temp(TF);
%         temp2(:,i)=temp;
%     end
%
%     Power_eeg_temp{stage+1}=temp2;
%
% end
%
% figure
% for gabor=1:32
%     subplot(4,8,gabor)
%     X=[Power_eeg_temp{1}(:,gabor); Power_eeg_temp{2}(:,gabor); Power_eeg_temp{3}(:,gabor); Power_eeg_temp{4}(:,gabor); Power_eeg_temp{5}(:,gabor)];
%     G=[Power_eeg_temp{1}(:,gabor)*0+0; Power_eeg_temp{2}(:,gabor)*0+1; Power_eeg_temp{3}(:,gabor)*0+2; Power_eeg_temp{4}(:,gabor)*0+3; Power_eeg_temp{5}(:,gabor)*0+4];
%
%     boxplot(X,G,'Symbol','rx','OutlierSize',4)
%     p=anova1(X,G,'off');
%     title([num2str(gabor) '  '  num2str(p)])
%
% end
%
%
%
% figure
% count=1;
% for gabor=[4 10 12 18 24 25]
%     subplot(2,3,count)
%     count=count+1;
%     X=[Power_eeg_temp{1}(:,gabor); Power_eeg_temp{2}(:,gabor); Power_eeg_temp{3}(:,gabor); Power_eeg_temp{4}(:,gabor); Power_eeg_temp{5}(:,gabor)];
%     G=[Power_eeg_temp{1}(:,gabor)*0+0; Power_eeg_temp{2}(:,gabor)*0+1; Power_eeg_temp{3}(:,gabor)*0+2; Power_eeg_temp{4}(:,gabor)*0+3; Power_eeg_temp{5}(:,gabor)*0+4];
%
%     boxplot(X,G)
%     xticklabels(stage_names)
%     title(num2str(gabor))
%
% end




records_margin=[1 3229 5658 8354 11320 14100 17060 19640 22370 24800 27790 30510 32680 35160 38280 length(real_label)];
temp_sample=1:length(real_label);

Power_eeg_temp={};
Power_eeg_temp_std={};
for stage=0:4
    temp=[];
    temp2=[];
    for i=1:32
        for rec=1:15
            temp=Power_eeg_pos((real_label==stage)&(temp_sample>records_margin(rec))'&(temp_sample<records_margin(rec+1))',i);
            TF = ~isoutlier(temp,'percentiles',[10 90]);
            temp=temp(TF);
            temp2(rec,i)=mean(temp);
        end
    end
    
    Power_eeg_temp{stage+1}=temp2;
    
end
sig_p=.005;
figure
tiledlayout(4,8);
for gabor=1:32
    nexttile
    X=[Power_eeg_temp{1}(:,gabor); Power_eeg_temp{2}(:,gabor); Power_eeg_temp{3}(:,gabor); Power_eeg_temp{4}(:,gabor); Power_eeg_temp{5}(:,gabor)];
    G=[Power_eeg_temp{1}(:,gabor)*0+0; Power_eeg_temp{2}(:,gabor)*0+1; Power_eeg_temp{3}(:,gabor)*0+2; Power_eeg_temp{4}(:,gabor)*0+3; Power_eeg_temp{5}(:,gabor)*0+4];
    
    boxplot(X,G,'Symbol','rx','OutlierSize',4)
    [p,~,stats]=anova1(X,G,'off');
    c = multcompare(stats,'Display','off');
    ylim_=ylim();
    title(num2str(gabor))
    hold on
    n_sig=sum(c(:,end)<sig_p);
    c_sig=find(c(:,end)<sig_p);
    if n_sig>0
        for i=1:n_sig
            plot([c(c_sig(i),1) c(c_sig(i),2)], [ ylim_(1)-i*(ylim_(2)-ylim_(1))/10 ylim_(1)-i*(ylim_(2)-ylim_(1))/10],'k')
            plot([c(c_sig(i),1)], [ ylim_(1)-i*(ylim_(2)-ylim_(1))/10 ],'k<')
            plot([c(c_sig(i),2)], [ ylim_(1)-i*(ylim_(2)-ylim_(1))/10],'k>')
        end
    end
    
    ylim([ylim_(1)-(n_sig+1)*(ylim_(2)-ylim_(1))/10 ylim_(2)])
    xticklabels(stage_names)
end



figure
tiledlayout(2,5);
count=1;
for gabor=[ 2 3 4 8 10 12 18  23 24 25]
    nexttile
    count=count+1;
    X=[Power_eeg_temp{1}(:,gabor); Power_eeg_temp{2}(:,gabor); Power_eeg_temp{3}(:,gabor); Power_eeg_temp{4}(:,gabor); Power_eeg_temp{5}(:,gabor)];
    G=[Power_eeg_temp{1}(:,gabor)*0+0; Power_eeg_temp{2}(:,gabor)*0+1; Power_eeg_temp{3}(:,gabor)*0+2; Power_eeg_temp{4}(:,gabor)*0+3; Power_eeg_temp{5}(:,gabor)*0+4];
    
    boxplot(X,G)
    xticklabels(stage_names)
    title(num2str(gabor))
    
    [p,~,stats]=anova1(X,G,'off');
    c = multcompare(stats,'Display','off');
    ylim_=ylim();
    hold on
    n_sig=sum(c(:,end)<sig_p);
    c_sig=find(c(:,end)<sig_p);
    if n_sig>0
        for i=1:n_sig
            plot([c(c_sig(i),1) c(c_sig(i),2)], [ ylim_(1)-i*(ylim_(2)-ylim_(1))/10 ylim_(1)-i*(ylim_(2)-ylim_(1))/10],'k')
            plot([c(c_sig(i),1)], [ ylim_(1)-i*(ylim_(2)-ylim_(1))/10 ],'k<')
            plot([c(c_sig(i),2)], [ ylim_(1)-i*(ylim_(2)-ylim_(1))/10],'k>')
        end
    end
    
    ylim([ylim_(1)-(n_sig+1)*(ylim_(2)-ylim_(1))/10 ylim_(2)])
    
    
end


figure
tiledlayout(2,5);
count=1;
for i=[ 2 3 4 8 10 12 18  23 24 25]
    nexttile
    plot((1:200)/100,gabor_eeg(i,:))
    
    
    ylabel('Amplitude')
    
    
    xlabel('Time (Sec)')
    
    ti={['(' num2str(i) ')']; ['f=' num2str(w(i),'%0.1f') '  u=' num2str(u(i),'%0.2f') '  s=' num2str(s(i),'%0.2f')]};
    title(ti)
    count=count+1;
end


figure
tiledlayout(2,5);
count=1;
for i=[ 2 3 4 8 10 12 18  23 24 25]
    
    nexttile
    ff=abs(fft(gabor_eeg(i,:)));
    ff=ff(1:101);
    plot(0:.5:50,ff)
    
    ylabel('|fft|')
    
    xlabel('Freq (Hz)')
    
    ti={['(' num2str(i) ')']; ['f=' num2str(w(i),'%0.1f') '  u=' num2str(u(i),'%0.2f') '  s=' num2str(s(i),'%0.2f')]};
    title(ti)
    count=count+1;
end





Power_eeg_temp={};
Power_eeg_temp_std={};
for stage=0:4
    temp=[];
    temp2=[];
    for i=1:32
        for rec=1:15
            temp=Power_eeg_pos((real_label==stage)&(temp_sample>records_margin(rec))'&(temp_sample<records_margin(rec+1))',i);
            TF = ~isoutlier(temp,'percentiles',[10 90]);
            temp=temp(TF);
            temp2(rec,i)=mean(temp);
        end
    end
    
    Power_eeg_temp{stage+1}=temp2;
    
end

%% EEG/EOG

label=[];
A=[];
for stage=0:4
    temp=[];
    
    for rec=1:15
        temp(rec)=mean(power_ratio((real_label==stage)&(temp_sample>records_margin(rec))'&(temp_sample<records_margin(rec+1))'));
        
    end
    
    TF = ~isoutlier(temp,'percentiles',[10 90]);
    temp=temp(TF);
    A=[A;temp'];
    label=[label;ones(length(temp),1)*stage];
end

sig_p=.05;
figure
boxplot(A,label)
ylabel('EEG effect / EOG effect')
xticklabels(stage_names)

[p,~,stats]=anova1(A,label,'off');
c = multcompare(stats,'Display','off');
ylim_=ylim();
hold on
n_sig=sum(c(:,end)<sig_p);
c_sig=find(c(:,end)<sig_p);
if n_sig>0
    for i=1:n_sig
        plot([c(c_sig(i),1) c(c_sig(i),2)], [ ylim_(1)-i*(ylim_(2)-ylim_(1))/10 ylim_(1)-i*(ylim_(2)-ylim_(1))/10],'k')
        plot([c(c_sig(i),1)], [ ylim_(1)-i*(ylim_(2)-ylim_(1))/10 ],'k<')
        plot([c(c_sig(i),2)], [ ylim_(1)-i*(ylim_(2)-ylim_(1))/10],'k>')
    end
end

ylim([ylim_(1)-(n_sig+1)*(ylim_(2)-ylim_(1))/10 ylim_(2)])




label=[];
A=[];
for stage=0:4
    temp=[];
    
    for rec=1:15
        temp(rec)=mean(EEG_power((real_label==stage)&(temp_sample>records_margin(rec))'&(temp_sample<records_margin(rec+1))'));
        
    end
    
    TF = ~isoutlier(temp,'percentiles',[10 90]);
    temp=temp(TF);
    A=[A;temp'];
    label=[label;ones(length(temp),1)*stage];
end

sig_p=.05;
figure
boxplot(A,label)
ylabel('EEG effect')
xticklabels(stage_names)

[p,~,stats]=anova1(A,label,'off');
c = multcompare(stats,'Display','off');
ylim_=ylim();
hold on
n_sig=sum(c(:,end)<sig_p);
c_sig=find(c(:,end)<sig_p);
if n_sig>0
    for i=1:n_sig
        plot([c(c_sig(i),1) c(c_sig(i),2)], [ ylim_(1)-i*(ylim_(2)-ylim_(1))/10 ylim_(1)-i*(ylim_(2)-ylim_(1))/10],'k')
        plot([c(c_sig(i),1)], [ ylim_(1)-i*(ylim_(2)-ylim_(1))/10 ],'k<')
        plot([c(c_sig(i),2)], [ ylim_(1)-i*(ylim_(2)-ylim_(1))/10],'k>')
    end
end

ylim([ylim_(1)-(n_sig+1)*(ylim_(2)-ylim_(1))/10 ylim_(2)])




label=[];
A=[];
for stage=0:4
    temp=[];
    
    for rec=1:15
        temp(rec)=mean(EOG_power((real_label==stage)&(temp_sample>records_margin(rec))'&(temp_sample<records_margin(rec+1))'));
        
    end
    
    TF = ~isoutlier(temp,'percentiles',[10 90]);
    temp=temp(TF);
    A=[A;temp'];
    label=[label;ones(length(temp),1)*stage];
end

sig_p=.05;
figure
boxplot(A,label)
ylabel('EOG effect')
xticklabels(stage_names)

[p,~,stats]=anova1(A,label,'off');
c = multcompare(stats,'Display','off');
ylim_=ylim();
hold on
n_sig=sum(c(:,end)<sig_p);
c_sig=find(c(:,end)<sig_p);
if n_sig>0
    for i=1:n_sig
        plot([c(c_sig(i),1) c(c_sig(i),2)], [ ylim_(1)-i*(ylim_(2)-ylim_(1))/10 ylim_(1)-i*(ylim_(2)-ylim_(1))/10],'k')
        plot([c(c_sig(i),1)], [ ylim_(1)-i*(ylim_(2)-ylim_(1))/10 ],'k<')
        plot([c(c_sig(i),2)], [ ylim_(1)-i*(ylim_(2)-ylim_(1))/10],'k>')
    end
end

ylim([ylim_(1)-(n_sig+1)*(ylim_(2)-ylim_(1))/10 ylim_(2)])


%% best for all stages




figure
Pow=mean(Power_eeg_pos,1);
N=32;
[a,b]=sort(Pow);
importance_i=b;
importance_v=a;



temp=zeros(32,1);
temp=importance_v(1:N);
label={};
for i =1:32
    label{i}=' ';
end
for i =1:N
    label{i}=num2str(importance_i(i));
end
pie(temp,label)

%% last
epoch=[4500,4258 , 4570];
figure

count=0;
for i =epoch
    
    load(['Inter_' num2str(i)])
    xtick_={};
    for k=1:2
        subplot(7,6,k+count*2)
        plot((1:200)/100-1,gabor_eeg(kernels(end-k+1)+1,:))
        xtick_{k}=kernels(end-k+1)+1;
        title(['(' num2str(kernels(end-k+1)+1) ')'])
    end
    
    subplot(7,1,count*2+2)
    imagesc((1:2800)*28/2800,1:2,(image_impact(1:2,:)).^.5)
    colormap jet
    yticks(1:2)
    yticklabels(xtick_)
    xticklabels({})
    ylabel('$\sqrt{eff}$','interpreter', 'latex')
    colorbar
    subplot(7,1,count*2+3)
    plot((1:2800)*28/2800,EEG)
    xlim([0 28])
    yticklabels({})
    if (count==2)
    xlabel('Time (Sec)')
    
    else
        xticklabels({})
    end
    colorbar
    
    count=count+1;
   
end




