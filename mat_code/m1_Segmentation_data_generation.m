
close all; clear all; clc; warning off;
cd(fileparts(mfilename('fullpath')));

%% Data Import
Datapath = '../data/test_case';  
Savepath = '../data';            
mkdir(Savepath)

zImgNorm = importdata([Datapath,filesep,'CEST.mat']);
mask = importdata([Datapath,filesep,'Mask', filesep, 'Mask_Brain.mat']);
mask_tumor = importdata([Datapath,filesep,'Mask', filesep, 'Mask_Ground_Truth.mat']);

%% Add Brain Mask
[xn, yn, zn] = size(zImgNorm); 
for z = 1:zn
    zImgNorm(:,:,z) = zImgNorm(:,:,z).*mask;
end

%% Segmentation Data Export
savedata = [];  savedata_mask = zeros(xn,yn);
for h = 1:xn
    for w = 1:yn
        if mask_tumor(h,w) == 1
            savedata_mask(h,w) = 1;    % Tumor
        elseif mask(h,w) == 1
            savedata_mask(h,w) = 0;    % Other brain tissues
        else
            savedata_mask(h,w) = 255;    % Background
        end
    end
end    
savedata = permute(zImgNorm, [3 1 2]);
save([Savepath,filesep,'Savedata.mat'],'savedata','savedata_mask');
