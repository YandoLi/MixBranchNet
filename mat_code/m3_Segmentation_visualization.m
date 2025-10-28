close all; clear all; clc; warning off;
cd(fileparts(mfilename('fullpath')));

Datapath = '../result/Segmentation'; 
Savepath = Datapath;

%% Data Import
M0image = importdata(['../data/test_case',filesep,'M0_Image.mat']);
mask = importdata(['../data/test_case/Mask', filesep, 'Mask_Brain.mat']);
ground_truth = importdata(['../data/test_case/Mask', filesep, 'Mask_Ground_Truth.mat']);

pred_data = importdata([Datapath, filesep, 'pred_mask.mat']);
pred_mask_post_processing = importdata([Datapath, filesep, 'pred_mask_post_processing.mat']);
labels = pred_data.labels;
pred_classes = pred_data.pred_classes;
pred_mask = logical(pred_classes).*mask;  
   
%% Evaluation metrics
% Dice socre
intersection = sum((pred_mask & ground_truth), 'all'); 
Dice = 2 * intersection / (sum(pred_mask(:)) + sum(ground_truth(:)));
% IOU
union = sum((pred_mask | ground_truth), 'all');
IOU = intersection / union;
% ACC, SEN, SPE, Pre
TP = sum(sum((mask == 1) & ((ground_truth == 1) & (pred_mask == 1))));  
TN = sum(sum((mask == 1) & ((ground_truth == 0) & (pred_mask == 0))));  
FP = sum(sum((mask == 1) & ((ground_truth == 0) & (pred_mask == 1))));  
FN = sum(sum((mask == 1) & ((ground_truth == 1) & (pred_mask == 0))));  
accuracy = (TP + TN) / (TP + TN + FP + FN);
sensitivity = TP / (TP + FN);
specificity = TN / (TN + FP);    
precision = TP / (TP + FP);

%% Visualization by Figure Plotting  
boundary = bwboundaries(ground_truth); 

[xn, yn] = size(M0image);
area_num0 = 1;  area_num1 = 1;  area_num2 = 1;
for h = 1:xn
    for w = 1:yn
        if mask(h,w) == 0 % Control
            area0(area_num0,1) = h;
            area0(area_num0,2) = w;
            area_num0 = area_num0 + 1;
        else
            if pred_classes(h,w) == 0 % Control
                area1(area_num1,1) = h;
                area1(area_num1,2) = w;
                area_num1 = area_num1 + 1;
            elseif pred_classes(h,w) == 1 % Tumor
                area2(area_num2,1) = h;
                area2(area_num2,2) = w;
                area_num2 = area_num2 + 1;
            end                             
        end
    end 
end

figure;
imshow(M0image, [], 'InitialMagnification', 'fit');clim([0,2000]);axis off;  
title('Tumor pred result', 'FontWeight', 'bold', 'FontSize', 14);                   
hold on;
if exist('area1', 'var')
plot(area1(:, 2), area1(:, 1), 'gs', 'MarkerSize', sqrt(2), 'MarkerFaceColor', 'g');
hold on;
end
if exist('area2', 'var')
    plot(area2(:, 2), area2(:, 1), 'rs', 'MarkerSize', sqrt(2), 'MarkerFaceColor', 'r');
    hold on;
end
for k = 1:length(boundary)
    plot(boundary{k}(:, 2), boundary{k}(:, 1), 'k--', 'LineWidth', 1);
end
dim = [0.3, 0.025, 0.4, 0.05]; 
str = sprintf('Dice Coefficient:  %.4f%%\nAccuracy: %.4f%%', Dice.*100, accuracy.*100);
annotation('textbox', dim, 'String', str, 'FitBoxToText', 'on', 'EdgeColor', 'none', 'BackgroundColor', 'none', 'Color', 'black', ...
    'FontWeight', 'bold','FontSize', 12, 'FontName', 'Times New Roman', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'Units', 'normalized');
hold off;
exportgraphics(gcf, [Datapath, filesep, 'pred_mask', '.png'], 'BackgroundColor', 'none');

figure;
imshow(pred_mask_post_processing, [],'InitialMagnification','fit');axis off;
exportgraphics(gcf, [Datapath, filesep, 'pred_mask_post_processing_1', '.png'], 'BackgroundColor', 'none');


colorMask = cat(3, 255.*pred_mask_post_processing, 255.*pred_mask_post_processing, 255.*pred_mask_post_processing);     
figure;
imshow(M0image, [],'InitialMagnification','fit');axis off;
hold on;
h = imshow(colorMask);
set(h, 'AlphaData', 0.50 * double(pred_mask_post_processing > 0)); 
hold on;  
for k = 1:length(boundary)
    plot(boundary{k}(:, 2), boundary{k}(:, 1), 'r--', 'LineWidth', 1);
end
hold off;
exportgraphics(gcf, [Datapath, filesep, 'pred_mask_post_processing_2', '.png'], 'BackgroundColor', 'none');


