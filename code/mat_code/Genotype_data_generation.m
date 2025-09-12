
close all; clear all; clc; warning off;

%% Data Import
Datapath = 'E:\yjs\paper\yando_tumorclassification_paper\构思论文架构相关材料\论文\JMRI\code\test_case';
Savepath = 'E:\yjs\paper\yando_tumorclassification_paper\构思论文架构相关材料\论文\JMRI\code\code\data';

mkdir(Savepath)
zImgNorm = importdata([Datapath,filesep,'CEST.mat']);
mask = importdata([Datapath,filesep,'Mask', filesep, 'Mask_Brain.mat']);
mask_tumor_pred = importdata(['E:\yjs\paper\yando_tumorclassification_paper\构思论文架构相关材料\论文\JMRI\code\result\segment',filesep,'pred_mask_post_processing.mat']);

savepath_idh = [Savepath,'\','IDH_patch'];
savepath_mgmt = [Savepath,'\','MGMT_patch'];
mkdir(savepath_idh)
mkdir(savepath_mgmt)

%% Add Brain Mask
[xn, yn, zn] = size(zImgNorm); 
for z = 1:zn
    zImgNorm(:,:,z) = zImgNorm(:,:,z).*mask;
end

%% IDH genotype Patch Data Export
savedata = [];  savedata_mask = [];
mask_label_idh = zeros(xn, yn);

for h = 1:xn
    for w = 1:yn
        if mask_tumor_pred(h,w) == 1
            % mask_label_idh(h,w) = 1;      % IDH wild: 1
            mask_label_idh(h,w) = 0;      % IDH mutation: 0
        else
            mask_label_idh(h,w) = 255;    % Background
        end
    end
end  

% Padding size
padSize = (256 - xn) / 2; 
zImgNorm_padding = padarray(zImgNorm, [padSize padSize 0], 0, 'both');    
mask_label_idh = padarray(mask_label_idh, [padSize padSize], 255, 'both');   

% Extract IDH Patch Blocks
patch_size = 64; stride = 16; patch_center = 48;
num_patches_x = floor((xn - patch_size) / stride) + 1; % patch number in row
num_patches_y = floor((yn - patch_size) / stride) + 1; % patch number in column
patch_num = 1;
patches = cell(num_patches_y, num_patches_x);          
for h = 1:num_patches_y
    for w = 1:num_patches_x
        x_start = (w-1) * stride + 1;
        x_end = x_start + patch_size - 1;
        y_start = (h-1) * stride + 1;
        y_end = y_start + patch_size - 1;
        savedata_mask(:,:) = mask_label_idh(y_start:y_end, x_start:x_end);
        for z = 1:zn
            savedata(z,:,:) = zImgNorm_padding(y_start:y_end, x_start:x_end, z);
        end
        save([savepath_idh,filesep,'patch_',num2str(h),'_',num2str(w),'.mat'],'savedata','savedata_mask');
        patch_num = patch_num + 1;
    end
end
%% MGMT genotype Patch Data Export
savedata = [];  savedata_mask = [];
mask_label_mgmt = zeros(xn, yn);

for h = 1:xn
    for w = 1:yn
        if mask_tumor_pred(h,w) == 1
            mask_label_mgmt(h,w) = 0;      % MGMT unmethylated : 0 
            % mask_label_mgmt(h,w) = 1;      % MGMT methylated : 1
        else
            mask_label_mgmt(h,w) = 255;    % Background
        end
    end
end  

% Extract MGMT Patch Blocks
patch_size = 48; stride = 12; patch_center = 24;
num_patches_x = floor((xn - patch_size) / stride) + 1; % patch number in row
num_patches_y = floor((yn - patch_size) / stride) + 1; % patch number in column

patch_num = 1;
patches = cell(num_patches_y, num_patches_x);        
for h = 1:num_patches_y
    for w = 1:num_patches_x
        x_start = (w-1) * stride + 1;
        x_end = x_start + patch_size - 1;
        y_start = (h-1) * stride + 1;
        y_end = y_start + patch_size - 1;
        savedata_mask(:,:) = mask_label_mgmt(y_start:y_end, x_start:x_end);
        for z = 1:zn
            savedata(z,:,:) = zImgNorm(y_start:y_end, x_start:x_end, z);
        end
        save([savepath_mgmt,filesep,'patch_',num2str(h),'_',num2str(w),'.mat'],'savedata','savedata_mask');
        patch_num = patch_num + 1;
    end
end