
close all; clear all; clc; warning off;
cd(fileparts(mfilename('fullpath')));

%% Data Import
genotype = 'MGMT'; % IDH or MGMT

Datapath = fullfile('../result', [genotype, '_genotype']);   

zImgNorm = importdata(['../data/test_case', filesep,'CEST.mat']);
M0image = importdata(['../data/test_case',filesep,'M0_Image.mat']);
mask = importdata(['../data/test_case/Mask',filesep,'Mask_Brain.mat']);
mask_tumor_pred = importdata(['..\result\Segmentation', filesep, 'pred_mask_post_processing.mat']);
pred = importdata([Datapath,filesep,'Result_Map.mat']);
pred_classes = pred.pred_classes;
pred_softmax = pred.pred_softmax;

pred_classes = logical(pred_classes);
pred_classes = pred_classes.*mask;

[xn, yn, zn] = size(zImgNorm); 

pred_softmax1 = zeros(xn, yn);
pred_softmax2 = zeros(xn, yn);

pred_softmax1(:,:) = pred_softmax(1,:,:);
pred_softmax1 = pred_softmax1.*mask_tumor_pred;
pred_softmax2(:,:) = pred_softmax(2,:,:);
pred_softmax2 = pred_softmax2.*mask_tumor_pred;

%% Patient-level Average Predicted Probability (PA) 
P1_avg2 = sum(sum(pred_softmax1)) / sum(sum(mask_tumor_pred));
P2_avg2 = sum(sum(pred_softmax2)) / sum(sum(mask_tumor_pred));
fprintf('Average P value(class 0): %.4f%%\n', P1_avg2.*100);
fprintf('Average P value(class 1): %.4f%%\n', P2_avg2.*100);

%% Classification Mask
mask_class_0 = false(240, 240); mask_class_1 = false(240, 240);
for h = 1:yn
    for w = 1:xn
        if mask_tumor_pred(h,w) == 1
            if pred_classes(h,w) == 0
                mask_class_0(h,w) = 1;  % class 0
            elseif pred_classes(h,w) == 1
                mask_class_1(h,w) = 1;  % class 1
            end
        end 
    end
end

%% Get Tumor Bounding Box (Window)
mask_tumor_size = size(mask);
h_max = 0;h_min = mask_tumor_size(1);
w_max = 0;w_min = mask_tumor_size(2);
for h0 = 1:1:mask_tumor_size(1)
    for w = 1:1:mask_tumor_size(2)
        if mask_tumor_pred(h0,w) == 1
            if h0>=h_max
                h_max = h0;
            end
            if h0<=h_min
                h_min = h0;
            end
            if w>=w_max
                w_max = w;
            end
            if w<=w_min
                w_min = w;
            end
        end
    end
end
h_max = h_max+10; h_min = h_min-10;
w_max = w_max+10; w_min = w_min-10;

window = zeros(xn, yn);
window(h_min:h_max,w_min:w_max) = 1;
window = logical(window);      

%% Visualization by Figure Plotting
boundary = bwboundaries(mask_tumor_pred);
boundary2 = bwboundaries(window);

min_x = min(boundary2{1}(:, 2));
max_x = max(boundary2{1}(:, 2));
min_y = min(boundary2{1}(:, 1));
max_y = max(boundary2{1}(:, 1)); 

switch genotype
    case 'MGMT'
        Colormask0 = cat(3, 1.*mask_class_0, 138.*mask_class_0, 103.*mask_class_0); 
        Colormask1 = cat(3, 243.*mask_class_1, 163.*mask_class_1, 50.*mask_class_1); 
    case 'IDH'
        Colormask0 = cat(3, 222.*mask_class_0, 88.*mask_class_0, 43.*mask_class_0); 
        Colormask1 = cat(3, 68.*mask_class_1, 114.*mask_class_1, 196.*mask_class_1); 
    otherwise
        error('Unknown genotype: %s', genotype);
end
Colormask0 = uint8(Colormask0); 
Colormask1 = uint8(Colormask1); 

figure;
imshow(M0image, [],'InitialMagnification','fit');axis off;
hold on;
h0 = imshow(Colormask0);
set(h0, 'AlphaData', 1.00 * double(mask_class_0 > 0));
h1 = imshow(Colormask1);
set(h1, 'AlphaData', 1.00 * double(mask_class_1 > 0));
hold on;  
boundary = bwboundaries(mask_tumor_pred); 
for k = 1:length(boundary)
    plot(boundary{k}(:, 2), boundary{k}(:, 1), 'k--', 'LineWidth', 1);
end
hold off;

exportgraphics(gcf, [Datapath, filesep,'Prediction_Map_1', '.png'], 'BackgroundColor', 'none');

figure;
imshow(M0image, [],'InitialMagnification','fit');axis off;
xlim([min_x, max_x]);
ylim([min_y, max_y]);
hold on;
h0 = imshow(Colormask0);
set(h0, 'AlphaData', 1.00 * double(mask_class_0 > 0));
h1 = imshow(Colormask1);
set(h1, 'AlphaData', 1.00 * double(mask_class_1 > 0));
hold on;
for k = 1:length(boundary)
    plot(boundary{k}(:, 2), boundary{k}(:, 1), 'k-', 'LineWidth', 1);
end
exportgraphics(gcf, [Datapath, filesep, 'Prediction_Map_2', '.png'], 'BackgroundColor', 'none');


figure; 
pred_softmax2_smooth = imgaussfilt(pred_softmax2, 0.5);
imshow(pred_softmax1.*mask_tumor_pred, [],'InitialMagnification','fit');axis off; % 显示矩阵
colormap('jet'); clim([0 1.01]);
exportgraphics(gcf, [Datapath, filesep, 'Probability_Map_1', '.png'], 'BackgroundColor', 'none');

figure; 
imshow(pred_softmax1.*mask_tumor_pred, [],'InitialMagnification','fit');axis off; % 显示矩阵
xlim([min_x, max_x]);
ylim([min_y, max_y]);
colormap('jet'); clim([0 1.01]);   
exportgraphics(gcf, [Datapath, filesep, 'Probability_Map_2', '.png'], 'BackgroundColor', 'none');
