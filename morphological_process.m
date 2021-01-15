close all; clear all; clc;


[data,path] = uigetfile('*.mat');
[ground_truth, path_gt] =  uigetfile('*.mat');

data = load(data);
ground_truth = load(ground_truth);

struct_data = fieldnames(data);
data = data.(struct_data{1});

struct_gt = fieldnames(ground_truth);
ground_truth = ground_truth.(struct_gt{1});

%[w, h, spektral] = size(data);
f = data(:,:,:);

subplot(1,2,1);
imshow(data(:,:,100),[]);
[x,y,z]=size(f);

se = strel('arbitrary',1);
for k = 1:1:z
    f = data(:,:,k);
 
    openData(:, :, k) = imopen(f, se); 
    
end

subplot(1,2,2);
imshow(openData(:, :,100), []);
title('Dilated image');

I =openData(:, :, 100)


