function [regression_label] = fast_rcnn_bbox_transform(ex_boxes, gt_boxes)
% [regression_label] = fast_rcnn_bbox_transform(ex_boxes, gt_boxes)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    ex_widths = ex_boxes(:, 3) - ex_boxes(:, 1) + 1;
    ex_heights = ex_boxes(:, 4) - ex_boxes(:, 2) + 1;
    ex_ctr_x = ex_boxes(:, 1) + 0.5 * (ex_widths - 1);
    ex_ctr_y = ex_boxes(:, 2) + 0.5 * (ex_heights - 1);
    
    gt_widths = gt_boxes(:, 3) - gt_boxes(:, 1) + 1;
    gt_heights = gt_boxes(:, 4) - gt_boxes(:, 2) + 1;
    gt_ctr_x = gt_boxes(:, 1) + 0.5 * (gt_widths - 1);
    gt_ctr_y = gt_boxes(:, 2) + 0.5 * (gt_heights - 1);
    
    targets_dx = (gt_ctr_x - ex_ctr_x) ./ (ex_widths+eps);
    targets_dy = (gt_ctr_y - ex_ctr_y) ./ (ex_heights+eps);
    targets_dw = log(gt_widths ./ ex_widths);
    targets_dh = log(gt_heights ./ ex_heights);
    
    regression_label = [targets_dx, targets_dy, targets_dw, targets_dh];
end