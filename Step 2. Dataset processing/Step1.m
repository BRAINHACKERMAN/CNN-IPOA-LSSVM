clc
clear
load data_total.mat
data = Rawdata(:,1:end-1);
%% file_path 一定要给文件路径
file_path ='同步提取小波变换图'; %
% 判断文件路径是否存在，如果文件路径不存在，则创建
if ~exist(file_path, 'dir')
    mkdir(file_path);
    fprintf('文件路径 %s 创建成功！\n', file_path);
end
%%  生成图像
% 计算 wset
for i=1:size(data,1)
    i
    xx = data(i,:); % 循环读取每个样本
    xx = xx';
    % 计算
    [sst,f]  = wset(xx);
    % 绘图
    imagesc(abs(sst));
    axis_control(flag)
    % 保存
    saveas(gcf,fullfile(file_path, [num2str(i) '.jpg'])) % 保存
end
