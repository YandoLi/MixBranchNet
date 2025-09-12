
close all; clear all; clc; warning off;

Rootpath = 'E:\yjs\paper\yando_tumorclassification_paper\构思论文架构相关材料\论文\JMRI\code';
Datapath = fullfile(Rootpath, 'code', 'result', 'Segmentation');
Savepath = Datapath;

%% Data Import
pred_data = importdata([Datapath, filesep, 'pred_mask.mat']);
labels = pred_data.labels;
pred_classes = pred_data.pred_classes;

pred_bin = logical(pred_classes);         
roi_mask = labels ~= 255;                

%% Post Processing
pred_bin(~roi_mask) = 0;                   % Shielding background

pred_clean = bwareaopen(pred_bin, 6);      % Delete small area

pred_filled = imfill(pred_clean, 'holes');  % Filling holes in ROI

pred_final = pred_filled & roi_mask;       

%% Data Export
pred_classes = double(pred_final);     
save([Savepath,filesep,'pred_mask_post_processing.mat'],'pred_classes');