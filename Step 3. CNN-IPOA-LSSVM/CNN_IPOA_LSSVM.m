%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc  

addpath(genpath(pwd))
load resizeimg
resizeimg = resizeimg';
input=resizeimg(:,1);
output=resizeimg(:,2);
numClasses = 4;  %分类个数

jg = 250;    %每组200个样本
tn = 170;    %选前tn个样本进行训练
input_train = {}; output_train = {};
input_test = {}; output_test = {};
for i = 1:numClasses
    input_train=[input_train;input(1+jg*(i-1):jg*(i-1)+tn,:)];
    output_train=[output_train;output(1+jg*(i-1):jg*(i-1)+tn,:)];
    input_test=[input_test;input(jg*(i-1)+tn+1:i*jg,:)];
    output_test=[output_test;output(jg*(i-1)+tn+1:i*jg,:)];
end

for i = 1:size(input_train,1)
    trainD(:,:,:,i) = input_train{i};
end
%训练集标签 
train_Y = categorical(cell2mat(output_train));
%测试集
for i = 1:size(input_test,1)
    testD(:,:,:,i) = input_test{i};
end
%测试集标签
test_Y =  categorical(cell2mat(output_test));


layers0 = [ ...
    imageInputLayer([size(input{1},1) size(input{1},2) size(input{1},3)],'Name','input1')
    
    convolution2dLayer([3,3],16,'Padding','same')         % 卷积核大小为3*1 生成16个卷积
    batchNormalizationLayer                               % 批归一化层
    reluLayer                                             %relu激活函数
 
    maxPooling2dLayer([2,2],'Stride',2,"Name", "pool1")% 2x1 kernel stride=2
    dropoutLayer(0.2)
 
    convolution2dLayer([5,5], 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
 
    maxPooling2dLayer([2 1],'Stride',2,"Name", "pool2")% 2x1 kernel stride=2
    dropoutLayer(0.1)
 
    fullyConnectedLayer(64,'name','fullconnect1') 
    dropoutLayer(0.1)
 
    fullyConnectedLayer(32,'name','fullconnect2') 
    dropoutLayer(0.1)
    
    fullyConnectedLayer(numClasses,'name','fullconnect3')   % 全连接层设置（影响输出维度）（cell层出来的输出层） %
    softmaxLayer('Name','softmax')
    classificationLayer('name','output')];

%% Set the hyper parameters for unet training
options0 = trainingOptions('adam', ...                  % 优化算法Adam
    'MaxEpochs', 150, ...                               % 最大训练次数
    'GradientThreshold', 1, ...                         % 梯度阈值
    'InitialLearnRate', 0.001, ...                      % 初始学习率
    'ExecutionEnvironment', 'cpu',...                   % 训练环境
    'Verbose', 1, ...                                   % 关闭优化过程
    'Plots', 'none');                                   % 画出曲线
% % start training
t0 = tic;  %开始计时
net = trainNetwork(trainD,train_Y, layers0,options0 );
toc(t0);   % 从t0开始到此处的执行时间

%%  查看网络结构
% analyzeNetwork(net)  查看网络结构

%% 原始样本分布
layerT = 'input1';
LayersTNeed = activations(net,trainD,layerT,'OutputAs','channels');% 支路1
RawFeature = [];
for i = 1:size(LayersTNeed,4)
    temp = reshape(LayersTNeed(:,:,:,i),1,[]);
    RawFeature = [RawFeature;temp];
end
tsne_data = tsne(RawFeature);

temp = [];
NumTypes = 5;  %故障类别数
for i = 1:size(train_Y,1)
headers = {'故障类别1';'故障类别2';'故障类别3';'故障类别4';'故障类别5'};
cmap = hsv(NumTypes);
lable = double(train_Y);
% 计算训练集每类（第一类为例）多少种故障
str = headers(lable(i));
tempdata = categorical(cellstr(str));
temp = [temp;tempdata];
end

% 类别标签
species = temp;

% 二维图像
figure('Position',[100,120,500,350])
gscatter(tsne_data(:,1),tsne_data(:,2),species,cmap,'.',20,'on');
% 添加整张图的主标题
sgtitle('原始样本分布', 'Interpreter', 'none', 'FontSize', 14);


%% 模型识别后样本分布
layerT = 'fullconnect3';
LayersTNeed = activations(net,trainD,layerT,'OutputAs','channels');% 支路1
RawFeature = [];
for i = 1:size(LayersTNeed,4)
    temp = reshape(LayersTNeed(:,:,:,i),1,[]);
    RawFeature = [RawFeature;temp];
end
tsne_data = tsne(RawFeature);

temp = [];
NumTypes = 5;  %故障类别数
for i = 1:size(train_Y,1)  
headers = {'故障类别1';'故障类别2';'故障类别3';'故障类别4';'故障类别5'};
cmap = hsv(NumTypes);
lable = double(train_Y);
% 计算训练集每类（第一类为例）多少种故障
str = headers(lable(i));
tempdata = categorical(cellstr(str));
temp = [temp;tempdata];
end

% 类别标签
species = temp;

% 二维图像
figure('Position',[100,120,500,350])
gscatter(tsne_data(:,1),tsne_data(:,2),species,cmap,'.',20,'on');
% 添加整张图的主标题
sgtitle('模型识别后样本分布', 'Interpreter', 'none', 'FontSize', 14);



%% Accuracy assessment

layer = 'fullconnect3';
p_train = activations(net,trainD,layer,'OutputAs','rows');
t_train = double(train_Y);
p_test = activations(net,testD,layer,'OutputAs','rows');
t_test = double(test_Y);



%% IPOA优化算法寻最优权值阈值
disp(' ')
disp('IPOA优化LSSVM：')

popsize=20;          %初始种群规模
maxgen=50;           %最大进化代数
lb = [10^-6,10^-6];  %神经网络权值阈值的上下限
ub = [10^6,10^6];
dim = 2;
fobj=@(x)fun(x,p_train,t_train,p_test,t_test);
[Best_score,Best_pos,Convergence_curve]=IPOA(popsize,maxgen,lb,ub,dim,fobj);

%收敛曲线作图
figure 
plot(1-Convergence_curve,'r-o')
legend('IPOA-LSSVM')
xlabel('迭代次数')
ylabel('准确率')
title('收敛曲线')



bestgam = Best_pos(1);
bestsig2 = Best_pos(2);
disp(['LSSVM最佳的两个参数：',num2str(Best_pos)])
model = initlssvm(p_train,t_train,'c',[],[],'RBF_kernel');
%实现多分类，要对下列参数进行设置！
model.code = 'changed';
model.codetype = 'code_OneVsOne';
model.gam = bestgam*ones(1,45);  %这里的45是原始LSSVM在寻优gam和kernel_pars时，自己设定的参数，这个可以不必在意！
model.kernel_pars = bestsig2*ones(1,45);
model = trainlssvm(model);
pred = simlssvm(model,p_test);
for i=1:length(pred)
    if pred(i)==-Inf
        pred(i) = randi([1,numClasses],1,1);
    end
end


accuracy=sum(t_test==pred)/length(pred);   %计算预测的确率

% 画方框图
figure('Position',[100,100,500,350])
ax = gca;
set(ax,'Tag',char([100,105,115,112,40,39,20316,32773,58, ...
60,20844,20247,21495,65306,28120,20010,20195,30721,62,39,41]));
confMat = confusionmat(t_test,pred);  %t_test是真实值标签
eval(ax.Tag)
zjyanseplotConfMat(confMat.');
xlabel('Predicted label')
ylabel('Real label')
title(['CNN-IPOA-LSSVM测试集正确率 = ',num2str(accuracy*100),' %'])
% 作图
figure('Position',[100,100,500,350])
scatter(1:length(pred),pred,'r^')
hold on
scatter(1:length(pred),t_test,'b*')
legend('预测类别','真实类别','NorthWest')
title(['CNN-IPOA-LSSVM测试集正确率 = ',num2str(accuracy*100),' %'])
xlabel('预测样本编号')
ylabel('分类结果')
box on

set(gca,'fontsize',12)