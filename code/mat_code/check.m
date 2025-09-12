% ini;
% % 路径（按需调整）
% file1 = 'E:\yjs\paper\yando_tumorclassification_paper\构思论文架构相关材料\论文\JMRI\code\code\data\Savedata.mat';
% file2 = 'E:\yjs\paper\yando_tumorclassification_paper\构思论文架构相关材料\论文\JMRI\code\code\data\test\Savedata.mat';
% 
% % 只加载需要检查的变量，避免无关干扰
% s1 = load(file1, 'savedata', 'savedata_mask');
% s2 = load(file2, 'savedata', 'savedata_mask');
% 
% % 比较 savedata
% isDataEqual = isequaln(s1.savedata, s2.savedata);   % isequaln 会把 NaN 视为相等
% % 比较 savedata_mask
% isMaskEqual = isequaln(s1.savedata_mask, s2.savedata_mask);
% 
% % 输出结果
% if isDataEqual && isMaskEqual
%     fprintf('✅ 两个 MAT 文件中的 savedata 和 savedata_mask 完全一致。\n');
% else
%     fprintf('❌ 两个 MAT 文件内容不一致：\n');
%     if ~isDataEqual
%         fprintf('- savedata 不一致。\n');
%         % 如果是数值矩阵，也可以输出差异统计，例如：
%         if isnumeric(s1.savedata) && isnumeric(s2.savedata) ...
%                 && isequal(size(s1.savedata), size(s2.savedata))
%             diffVal = s1.savedata - s2.savedata;
%             fprintf('  savedata 最大绝对差值 = %.6g\n', max(abs(diffVal(:))));
%         end
%     end
%     if ~isMaskEqual
%         fprintf('- savedata_mask 不一致。\n');
%         if isnumeric(s1.savedata_mask) && isnumeric(s2.savedata_mask) ...
%                 && isequal(size(s1.savedata_mask), size(s2.savedata_mask))
%             diffMask = s1.savedata_mask - s2.savedata_mask;
%             fprintf('  savedata_mask 最大绝对差值 = %.6g\n', max(abs(diffMask(:))));
%         end
%     end
% end
% ini
% % 路径设置
% dir1 = 'E:\yjs\paper\yando_tumorclassification_paper\构思论文架构相关材料\论文\JMRI\code\code\data\IDH_patch';
% dir2 = 'E:\yjs\paper\yando_tumorclassification_paper\构思论文架构相关材料\论文\JMRI\code\code\data\test\IDH_patch';
% 
% % 循环比较所有 x=1~17, y=1~17
% for x = 1:12
%     for y = 1:12
%         % 构造文件名
%         file1 = fullfile(dir1, sprintf('patch_%d_%d_zFit.mat', x, y));
%         file2 = fullfile(dir2, sprintf('patch_%d_%d.mat', x, y));
% 
%         % 检查文件是否存在
%         if ~isfile(file1)
%             fprintf('⚠️ 缺少文件: %s\n', file1);
%             continue;
%         end
%         if ~isfile(file2)
%             fprintf('⚠️ 缺少文件: %s\n', file2);
%             continue;
%         end
% 
%         % 加载两个文件的 savedata 和 savedata_mask
%         s1 = load(file1, 'savedata', 'savedata_mask');
%         s2 = load(file2, 'savedata', 'savedata_mask');
% 
%         % 比较是否相同
%         isDataEqual = isequaln(s1.savedata, s2.savedata);
%         isMaskEqual = isequaln(s1.savedata_mask, s2.savedata_mask);
% 
%         if isDataEqual && isMaskEqual
%             fprintf('✅ patch_%d_%d 完全一致\n', x, y);
%         else
%             fprintf('❌ patch_%d_%d 不一致:\n', x, y);
%             if ~isDataEqual
%                 fprintf('   - savedata 不同\n');
%                 if isnumeric(s1.savedata) && isnumeric(s2.savedata) && ...
%                    isequal(size(s1.savedata), size(s2.savedata))
%                     fprintf('     最大差值: %.6g\n', max(abs(s1.savedata(:) - s2.savedata(:))));
%                 end
%             end
%             if ~isMaskEqual
%                 fprintf('   - savedata_mask 不同\n');
%                 if isnumeric(s1.savedata_mask) && isnumeric(s2.savedata_mask) && ...
%                    isequal(size(s1.savedata_mask), size(s2.savedata_mask))
%                     fprintf('     最大差值: %.6g\n', max(abs(s1.savedata_mask(:) - s2.savedata_mask(:))));
%                 end
%             end
%         end
%     end
% end

% ini;
% % 路径
% file1 = 'E:\yjs\paper\yando_tumorclassification_paper\构思论文架构相关材料\论文\JMRI\code\result\genotype_prediction\MGMT\Result_Map.mat';
% file2 = 'E:\yjs\paper\yando_tumorclassification_paper\构思论文架构相关材料\论文\JMRI\code\code\result\MGMT_genotype\Result_Map.mat';
% 
% % 载入.mat文件
% try
%     data1 = load(file1, 'pred_classes', 'pred_softmax');
%     data2 = load(file2, 'pred_classes', 'pred_softmax');
% catch ME
%     disp(['Error loading files: ', ME.message]);
%     return;
% end
% 
% % 比较 pred_classes
% isClassesEqual = isequaln(data1.pred_classes, data2.pred_classes);  % 允许 NaN 相等
% if isClassesEqual
%     disp('pred_classes is identical in both files.');
% else
%     disp('pred_classes is different between the two files.');
% end
% 
% % 比较 pred_softmax
% isSoftmaxEqual = isequaln(data1.pred_softmax, data2.pred_softmax);  % 允许 NaN 相等
% if isSoftmaxEqual
%     disp('pred_softmax is identical in both files.');
% else
%     disp('pred_softmax is different between the two files.');
% end
% 
% % 输出差异 (如果有)
% if ~isClassesEqual || ~isSoftmaxEqual
%     % 比较数值差异 (如果不是完全相等)
%     if ~isClassesEqual
%         fprintf('Difference in pred_classes:\n');
%         diff_classes = data1.pred_classes - data2.pred_classes;
%         max_diff_classes = max(abs(diff_classes(:)));
%         fprintf('Max difference in pred_classes: %.6f\n', max_diff_classes);
%     end
%     if ~isSoftmaxEqual
%         fprintf('Difference in pred_softmax:\n');
%         diff_softmax = data1.pred_softmax - data2.pred_softmax;
%         max_diff_softmax = max(abs(diff_softmax(:)));
%         fprintf('Max difference in pred_softmax: %.6f\n', max_diff_softmax);
%     end
% end

ini;
% 路径
file1 = 'E:\yjs\paper\yando_tumorclassification_paper\构思论文架构相关材料\论文\JMRI\code\code\result\Segmentation\pred_mask_post_processing.mat';
file2 = 'E:\yjs\paper\yando_tumorclassification_paper\构思论文架构相关材料\论文\JMRI\code\code\result\Segmentation\pred_mask_post.mat';

% 载入.mat文件
try
    data1 = load(file1, 'pred_classes');
    data2 = load(file2, 'pred_classes');
catch ME
    disp(['Error loading files: ', ME.message]);
    return;
end

% 比较 pred_classes
isClassesEqual = isequaln(data1.pred_classes, data2.pred_classes);  % 允许 NaN 相等
if isClassesEqual
    disp('pred_classes is identical in both files.');
else
    disp('pred_classes is different between the two files.');
end


% 输出差异 (如果有)
if ~isClassesEqual
    % 比较数值差异 (如果不是完全相等)
    if ~isClassesEqual
        fprintf('Difference in pred_classes:\n');
        diff_classes = data1.pred_classes - data2.pred_classes;
        max_diff_classes = max(abs(diff_classes(:)));
        fprintf('Max difference in pred_classes: %.6f\n', max_diff_classes);
    end
end