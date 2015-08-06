function proposal_visual_debug(conf, image_roidb, input_blobs, bbox_means, bbox_stds, classes, scale_inds)
% proposal_visual_debug(conf, image_roidb, input_blobs, bbox_means, bbox_stds, classes, scale_inds)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   

    im_blob = input_blobs{1};
    labels_blob = input_blobs{2};
    label_weights_blob = input_blobs{3};
    bbox_targets_blob = input_blobs{4};
    bbox_loss_weights_blob = input_blobs{5};
    
    % concat anchors as [channel, height, width], where channel is the fastest dimension.
    rois = proposal_locate_anchors(conf, image_roidb.im_size, conf.scales(scale_inds));

    bbox_targets = double(permute(bbox_targets_blob, [3, 2, 1]));
    bbox_targets = reshape(bbox_targets, 4, [])';
    bbox_targets = bsxfun(@times, bbox_targets, bbox_stds);
    bbox_targets = bsxfun(@plus, bbox_targets, bbox_means);
    
    labels_blob = double(permute(labels_blob, [3, 2, 1]));
    labels_blob = labels_blob(:);
    label_weights_blob = double(permute(label_weights_blob, [3, 2, 1]));
    label_weights_blob = label_weights_blob(:);
    pred_boxes = fast_rcnn_bbox_transform_inv(rois, bbox_targets);    

    num_anchors = size(conf.anchors, 1);
    for i = 1:size(im_blob, 4)
       for j = 1:num_anchors
           im = im_blob(:, :, [3, 2, 1], i);
           im = permute(im, [2, 1, 3]);
           imshow(mat2gray(im));
           hold on;
           
           sub_rois = rois(j:num_anchors:end, :);
           sub_labels = labels_blob(j:num_anchors:end);
           sub_label_weights = label_weights_blob(j:num_anchors:end);
           sub_pred_boxes = pred_boxes(j:num_anchors:end, :);
           
           % bg
           bg_ind = find(sub_labels == 0 & sub_label_weights > 0);
           if ~isempty(bg_ind)
               cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'k'), ...
                   num2cell(sub_rois(bg_ind, :), 2));
               cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'b'), ...
                   num2cell(sub_rois(bg_ind(round(length(bg_ind)/2)), :), 2));
           end
           
           % fg
           fg_ind = sub_labels > 0;
           cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'r'), ...
               num2cell(sub_rois(fg_ind, :), 2));
           cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'g'), ...
               num2cell(sub_pred_boxes(fg_ind, :), 2));
           
%            % others
%            others_ind = find(sub_labels == 0 & sub_label_weights == 0);
%            cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', [0.5, 0.5, 0.5]), ...
%                num2cell(sub_rois(others_ind, :), 2));

           hold off;   
           pause;
       end
    end
end