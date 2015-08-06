function [pred_boxes] = fast_rcnn_bbox_transform_inv(boxes, box_deltas)
% [pred_boxes] = fast_rcnn_bbox_transform_inv(boxes, box_deltas)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
    
    src_w = double(boxes(:, 3) - boxes(:, 1) + 1);
    src_h = double(boxes(:, 4) - boxes(:, 2) + 1);
    src_ctr_x = double(boxes(:, 1) + 0.5*(src_w-1));
    src_ctr_y = double(boxes(:, 2) + 0.5*(src_h-1));
    
    dst_ctr_x = double(box_deltas(:, 1:4:end));
    dst_ctr_y = double(box_deltas(:, 2:4:end));
    dst_scl_x = double(box_deltas(:, 3:4:end));
    dst_scl_y = double(box_deltas(:, 4:4:end));

    pred_ctr_x = bsxfun(@plus, bsxfun(@times, dst_ctr_x, src_w), src_ctr_x);
    pred_ctr_y = bsxfun(@plus, bsxfun(@times, dst_ctr_y, src_h), src_ctr_y);
    pred_w = bsxfun(@times, exp(dst_scl_x), src_w);
    pred_h = bsxfun(@times, exp(dst_scl_y), src_h);
    pred_boxes = zeros(size(box_deltas), 'single');
    pred_boxes(:, 1:4:end) = pred_ctr_x - 0.5*(pred_w-1);
    pred_boxes(:, 2:4:end) = pred_ctr_y - 0.5*(pred_h-1);
    pred_boxes(:, 3:4:end) = pred_ctr_x + 0.5*(pred_w-1);
    pred_boxes(:, 4:4:end) = pred_ctr_y + 0.5*(pred_h-1); 
end