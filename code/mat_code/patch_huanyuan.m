
close all; clear all; clc; warning off;

%% Parameter Setting and Data Initialization
genotype = 'IDH';

switch genotype
    case 'MGMT'
        image_size = 240;
        patch_size = 48;
        stride     = 12;
    case 'IDH'
        image_size = 256;
        patch_size = 64;
        stride     = 16;
    otherwise
        error('Unknown genotype: %s', genotype);
end

t = (patch_size - stride) / 2 ; 

% Datapath = fullfile('E:\yjs\paper\yando_tumorclassification_paper\构思论文架构相关材料\论文\JMRI\code\code\result', [genotype, '_genotype'],'result');
% Savepath = fullfile('E:\yjs\paper\yando_tumorclassification_paper\构思论文架构相关材料\论文\JMRI\code\code\result', [genotype, '_genotype']);    

Datapath = 'D:\lab data\patch\CEST_Classify\IDH_result\patch_64_stride16\IDH0_3\result';
Savepath = 'D:\lab data\patch\CEST_Classify\IDH_result\patch_64_stride16\IDH0_3';  

pred_classes_full = zeros(image_size, image_size);
pred_softmax_full = zeros(2, image_size, image_size); 
patch_num = floor((240 - patch_size) / stride) + 1; % patch number in row/column

%% Data Import
for h = 1:patch_num
    for w = 1:patch_num
        file_name = sprintf('patch_%d_%d_pred.mat', h, w);
        file_path = fullfile(Datapath, file_name);
        data = load(file_path);  
        pred_classes_patch = data.pred_classes; 
        pred_softmax_patch = data.pred_softmax; 
%% Patch Stitching 
        % Crop Center Regions
        start_idx = (patch_size - stride) / 2 + 1;
        end_idx = start_idx + stride -1;  

        cropped_pred_classes = pred_classes_patch(start_idx:end_idx, start_idx:end_idx);
        cropped_pred_softmax = pred_softmax_patch(:, start_idx:end_idx, start_idx:end_idx);

        x_start = (w-1) * stride + 1 + t; 
        x_end = w * stride + t;
        y_start = (h-1) * stride + 1 + t;
        y_end = h * stride + t;
        
        pred_classes_full(y_start:y_end, x_start:x_end) = cropped_pred_classes;     
        pred_softmax_full(:, y_start:y_end, x_start:x_end) = cropped_pred_softmax;  
    end
end

%% Iamge Crop
if image_size ~= 240     
    crop_start = floor((image_size - 240) / 2) + 1;
    crop_end = crop_start + 240 - 1;

    pred_classes_cropped = pred_classes_full(crop_start:crop_end, crop_start:crop_end);    
    pred_softmax_cropped = pred_softmax_full(:, crop_start:crop_end, crop_start:crop_end);

    pred_classes = pred_classes_cropped;
    pred_softmax = pred_softmax_cropped;
else
    pred_classes = pred_classes_full;
    pred_softmax = pred_softmax_full;    
end

%% Data Export
save([Savepath,filesep,'Result_Map.mat'],'pred_classes','pred_softmax');


