function [pred_boxes, scores] = fast_rcnn_conv_feat_detect(conf, caffe_net, im, conv_feat_blob, boxes, max_rois_num_in_gpu)
% [pred_boxes, scores] = fast_rcnn_conv_feat_detect(conf, caffe_net, im, conv_feat_blob, boxes, max_rois_num_in_gpu)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    [rois_blob, ~] = get_blobs(conf, im, boxes);
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    rois_blob = rois_blob - 1; % to c's index (start from 0)
    rois_blob = permute(rois_blob, [3, 4, 2, 1]);
    rois_blob = single(rois_blob);
    
    % set conv feature map as 'data'
    caffe_net.blobs('data').copy_data_from(conv_feat_blob);
    
    total_rois = size(rois_blob, 4);
    total_scores = cell(ceil(total_rois / max_rois_num_in_gpu), 1);
    total_box_deltas = cell(ceil(total_rois / max_rois_num_in_gpu), 1);
    for i = 1:ceil(total_rois / max_rois_num_in_gpu)
        
        sub_ind_start = 1 + (i-1) * max_rois_num_in_gpu;
        sub_ind_end = min(total_rois, i * max_rois_num_in_gpu);
        sub_rois_blob = rois_blob(:, :, :, sub_ind_start:sub_ind_end);
        
        % only set rois blob here
        net_inputs = {[], sub_rois_blob};

        % Reshape net's input blobs
        caffe_net.reshape_as_input(net_inputs);
        output_blobs = caffe_net.forward(net_inputs);

        if conf.test_binary
            % simulate binary logistic regression
            scores = caffe_net.blobs('cls_score').get_data();
            scores = squeeze(scores)';
            % Return scores as fg - bg
            scores = bsxfun(@minus, scores, scores(:, 1));
        else
            % use softmax estimated probabilities
            scores = output_blobs{2};
            scores = squeeze(scores)';
        end

        % Apply bounding-box regression deltas
        box_deltas = output_blobs{1};
        box_deltas = squeeze(box_deltas)';
        
        total_scores{i} = scores;
        total_box_deltas{i} = box_deltas;
    end 
    
    scores = cell2mat(total_scores);
    box_deltas = cell2mat(total_box_deltas);
    
    pred_boxes = fast_rcnn_bbox_transform_inv(boxes, box_deltas);
    pred_boxes = clip_boxes(pred_boxes, size(im, 2), size(im, 1));
    
    % remove scores and boxes for back-ground
    pred_boxes = pred_boxes(:, 5:end);
    scores = scores(:, 2:end);
end

function [rois_blob, im_scale_factors] = get_blobs(conf, im, rois)
    im_scale_factors = get_image_blob_scales(conf, im);
    rois_blob = get_rois_blob(conf, rois, im_scale_factors);
end

function im_scales = get_image_blob_scales(conf, im)
    im_scales = arrayfun(@(x) prep_im_for_blob_size(size(im), x, conf.test_max_size), conf.test_scales, 'UniformOutput', false);
    im_scales = cell2mat(im_scales); 
end

function [rois_blob] = get_rois_blob(conf, im_rois, im_scale_factors)
    [feat_rois, levels] = map_im_rois_to_feat_rois(conf, im_rois, im_scale_factors);
    rois_blob = single([levels, feat_rois]);
end

function [feat_rois, levels] = map_im_rois_to_feat_rois(conf, im_rois, scales)
    im_rois = single(im_rois);
    
    if length(scales) > 1
        widths = im_rois(:, 3) - im_rois(:, 1) + 1;
        heights = im_rois(:, 4) - im_rois(:, 2) + 1;
        
        areas = widths .* heights;
        scaled_areas = bsxfun(@times, areas(:), scales(:)'.^2);
        levels = max(abs(scaled_areas - 224.^2), 2); 
    else
        levels = ones(size(im_rois, 1), 1);
    end
    
    feat_rois = round(bsxfun(@times, im_rois-1, scales(levels))) + 1;
end

function boxes = clip_boxes(boxes, im_width, im_height)
    % x1 >= 1 & <= im_width
    boxes(:, 1:4:end) = max(min(boxes(:, 1:4:end), im_width), 1);
    % y1 >= 1 & <= im_height
    boxes(:, 2:4:end) = max(min(boxes(:, 2:4:end), im_height), 1);
    % x2 >= 1 & <= im_width
    boxes(:, 3:4:end) = max(min(boxes(:, 3:4:end), im_width), 1);
    % y2 >= 1 & <= im_height
    boxes(:, 4:4:end) = max(min(boxes(:, 4:4:end), im_height), 1);
end
    