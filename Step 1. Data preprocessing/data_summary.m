clc;
clear;
addpath(genpath(pwd))
load data.mat
numm = 3; %选取第三列数据，如果需要改，就改这里的numm即可！
data1=csv_acc_data{11,3}(1000:end,numm);  %从1000开始，是因为去除断点效应
data2=csv_acc_data{13,3}(1000:end,numm);
data3=csv_acc_data{15,3}(1000:end,numm);
data4=csv_acc_data{17,3}(1000:end,numm);
data5=csv_acc_data{19,3}(1000:end,numm);

% 一共是5个状态，每个状态有m组样本，每个样本的数据量大小为：1×s
w=1000;                  % w是滑动窗口的步长为1000
s=2048;                  % 滑动窗口大小，表示每个样本有2048个点
m = 200;                 %每种故障设置200个样本

D1=[];
for i =1:m
    D1 = [D1,data1(1+w*(i-1):w*(i-1)+s)];
end
D1 = D1';

D2=[];
for i =1:m
    D2 = [D2,data2(1+w*(i-1):w*(i-1)+s)];
end
D2 = D2';
D3=[];
for i =1:m
    D3 = [D3,data3(1+w*(i-1):w*(i-1)+s)];
end
D3 = D3';
D4=[];
for i =1:m
    D4 = [D4,data4(1+w*(i-1):w*(i-1)+s)];
end
D4 = D4';
D5=[];
for i =1:m
    D5 = [D5,data5(1+w*(i-1):w*(i-1)+s)];
end
D5 = D5';
data = [D1;D2;D3;D4;D5];



% 数据载入
bv = m;    %每种状态数据有m组
% 加标签值
hhh = size(data,2);
for i=1:size(data,1)/bv
    data(1+bv*(i-1):bv*i,hhh+1)=i;
end

Rawdata = data;
save data_total Rawdata



rmpath(pwd)
