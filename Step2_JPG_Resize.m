%重新调整图像大小，否则图像过大，不利于训练！
clc
clear all
load data_total.mat      %这里加载原数据，只是为了获取标签列
label = Rawdata(:,end);  %取出标签列

inputFolder = '同步提取小波变换图'; 


% 获取文件夹中所有的图像文件
imageFiles = dir(fullfile(inputFolder, '*.jpg'));

% 提取文件名中的数字部分
imageNumbers = cellfun(@(x) sscanf(x, '%d.png'), {imageFiles.name});

% 按照数字部分排序
[sortedNumbers, sortOrder] = sort(imageNumbers);

% 根据排序后的索引处理图像文件
sortedImageFiles = imageFiles(sortOrder);
numImages = length(sortedImageFiles);

imageSize = [64,64,3]; % 目标图像尺寸

for i = 1:numImages
    i
    imageFileName = sortedImageFiles(i).name;
    imagePath = fullfile(inputFolder, imageFileName);
    % 读取图像
    img = imread(imagePath);
    % 调整图像尺寸
    imgResized = imresize(img, imageSize(1:2)); 
    imgg = double(imgResized)/255;
    resizeimg{1,i} = imgg;
    resizeimg{2,i} = label(i);
end

save resizeimg.mat resizeimg
disp('处理完成！');
