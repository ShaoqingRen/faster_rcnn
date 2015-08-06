function [image_roidb, bbox_means, bbox_stds] = proposal_prepare_image_roidb(conf, imdbs, roidbs, bbox_means, bbox_stds)
% [image_roidb, bbox_means, bbox_stds] = proposal_prepare_image_roidb(conf, imdbs, roidbs, cache_img, bbox_means, bbox_stds)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   

    if ~exist('bbox_means', 'var')
        bbox_means = [];
        bbox_stds = [];
    end
    
    if ~iscell(imdbs)
        imdbs = {imdbs};
        roidbs = {roidbs};
    end

    imdbs = imdbs(:);
    roidbs = roidbs(:);
    
    if conf.target_only_gt
        image_roidb = ...
            cellfun(@(x, y) ... // @(imdbs, roidbs)
                arrayfun(@(z) ... //@([1:length(x.image_ids)])
                    struct('image_path', x.image_at(z), 'image_id', x.image_ids{z}, 'im_size', x.sizes(z, :), 'imdb_name', x.name, 'num_classes', x.num_classes, ...
                    'boxes', y.rois(z).boxes(y.rois(z).gt, :), 'class', y.rois(z).class(y.rois(z).gt, :), 'image', [], 'bbox_targets', []), ...
                [1:length(x.image_ids)]', 'UniformOutput', true),...
            imdbs, roidbs, 'UniformOutput', false);
    else
        image_roidb = ...
            cellfun(@(x, y) ... // @(imdbs, roidbs)
                arrayfun(@(z) ... //@([1:length(x.image_ids)])
                    struct('image_path', x.image_at(z), 'image_id', x.image_ids{z}, 'im_size', x.sizes(z, :), 'imdb_name', x.name, ...
                    'boxes', y.rois(z).boxes, 'class', y.rois(z).class, 'image', [], 'bbox_targets', []), ...
                [1:length(x.image_ids)]', 'UniformOutput', true),...
            imdbs, roidbs, 'UniformOutput', false);
    end
   
    image_roidb = cat(1, image_roidb{:});
    
    % enhance roidb to contain bounding-box regression targets
    [image_roidb, bbox_means, bbox_stds] = append_bbox_regression_targets(conf, image_roidb, bbox_means, bbox_stds);
end

function [image_roidb, means, stds] = append_bbox_regression_targets(conf, image_roidb, means, stds)
    % means and stds -- (k+1) * 4, include background class
    
    num_images = length(image_roidb);
    % Infer number of classes from the number of columns in gt_overlaps
    image_roidb_cell = num2cell(image_roidb, 2);
    bbox_targets = cell(num_images, 1);
    parfor i = 1:num_images
        % for fcn, anchors are concated as [channel, height, width], where channel is the fastest dimension.
       [anchors, im_scales] = proposal_locate_anchors(conf, image_roidb_cell{i}.im_size);
        
       gt_rois = image_roidb_cell{i}.boxes;
       gt_labels = image_roidb_cell{i}.class;
       im_size = image_roidb_cell{i}.im_size;
       bbox_targets{i} = cellfun(@(x, y) ...
           compute_targets(conf, scale_rois(gt_rois, im_size, y), gt_labels,  x, image_roidb_cell{i}, y), ...
           anchors, im_scales, 'UniformOutput', false);
    end
    clear image_roidb_cell;
    for i = 1:num_images
        image_roidb(i).bbox_targets = bbox_targets{i};
    end
    clear bbox_targets;
    
    if ~(exist('means', 'var') && ~isempty(means) && exist('stds', 'var') && ~isempty(stds))
        % Compute values needed for means and stds
        % var(x) = E(x^2) - E(x)^2
        class_counts = zeros(1, 1) + eps;
        sums = zeros(1, 4);
        squared_sums = zeros(1, 4);
        for i = 1:num_images
           for j = 1:length(conf.scales)
                targets = image_roidb(i).bbox_targets{j};
                gt_inds = find(targets(:, 1) > 0);
                if ~isempty(gt_inds)
                    class_counts = class_counts + length(gt_inds); 
                    sums = sums + sum(targets(gt_inds, 2:end), 1);
                    squared_sums = squared_sums + sum(targets(gt_inds, 2:end).^2, 1);
                end
           end
        end

        means = bsxfun(@rdivide, sums, class_counts);
        stds = (bsxfun(@minus, bsxfun(@rdivide, squared_sums, class_counts), means.^2)).^0.5;
    end
    
    % Normalize targets
    for i = 1:num_images
        for j = 1:length(conf.scales)
            targets = image_roidb(i).bbox_targets{j};
            gt_inds = find(targets(:, 1) > 0);
            if ~isempty(gt_inds)
                image_roidb(i).bbox_targets{j}(gt_inds, 2:end) = ...
                    bsxfun(@minus, image_roidb(i).bbox_targets{j}(gt_inds, 2:end), means);
                image_roidb(i).bbox_targets{j}(gt_inds, 2:end) = ...
                    bsxfun(@rdivide, image_roidb(i).bbox_targets{j}(gt_inds, 2:end), stds);
            end
        end
    end
end

function scaled_rois = scale_rois(rois, im_size, im_scale)
    im_size_scaled = round(im_size * im_scale);
    scale = (im_size_scaled - 1) ./ (im_size - 1);
    scaled_rois = bsxfun(@times, rois-1, [scale(2), scale(1), scale(2), scale(1)]) + 1;
end

function bbox_targets = compute_targets(conf, gt_rois, gt_labels, ex_rois, image_roidb, im_scale)
% output: bbox_targets
%   positive: [class_label, regression_label]
%   ingore: [0, zero(regression_label)]
%   negative: [-1, zero(regression_label)]

    if isempty(gt_rois)
        bbox_targets = zeros(size(ex_rois, 1), 5, 'double');
        bbox_targets(:, 1) = -1;
        return;
    end

    % ensure gt_labels is in single
    gt_labels = single(gt_labels);
    assert(all(gt_labels > 0));

    ex_gt_overlaps = boxoverlap(ex_rois, gt_rois);
    if conf.drop_boxes_runoff_image
         contained_in_image = is_contain_in_image(ex_rois, round(image_roidb.im_size * im_scale));
         ex_gt_overlaps(~contained_in_image, :) = 0;
    end

    [ex_max_overlaps, ex_assignment] = max(ex_gt_overlaps, [], 2);
    
    % assign a rois for each gt at least
    [gt_max_overlaps, gt_assignment] = max(ex_gt_overlaps, [], 1);
    
    % gt's best matches
    [gt_best_matches, gt_ind] = find(bsxfun(@eq, ex_gt_overlaps, [gt_max_overlaps]));
    % gt_best_matches_overlaps = ex_gt_overlaps(sub2ind(size(ex_gt_overlaps), gt_best_matches, gt_ind));
    
    % Indices of examples for which we try to make predictions
    fg_inds = unique([find(ex_max_overlaps >= conf.fg_thresh); gt_best_matches]);
        
    % Indices of examples for which we try to used as negtive samples
    bg_inds = setdiff(find(ex_max_overlaps < conf.bg_thresh_hi & ex_max_overlaps >= conf.bg_thresh_lo), ...
                    fg_inds);
    
    if conf.drop_boxes_runoff_image
        contained_in_image_ind = find(contained_in_image);
        fg_inds = intersect(fg_inds, contained_in_image_ind);
        bg_inds = intersect(bg_inds, contained_in_image_ind);
    end
                
    % Find which gt ROI each ex ROI has max overlap with:
    % this will be the ex ROI's gt target
    target_rois = gt_rois(ex_assignment(fg_inds), :);
    src_rois = ex_rois(fg_inds, :);
    
    [regression_label] = fast_rcnn_bbox_transform(src_rois, target_rois);
    
    bbox_targets = zeros(size(ex_rois, 1), 5, 'double');
    bbox_targets(fg_inds, :) = [gt_labels(ex_assignment(fg_inds)), regression_label];
    bbox_targets(bg_inds, 1) = -1;
    
    if 0 % debug
        %%%%%%%%%%%%%%
        im = imread(image_roidb.image_path);
        [im, im_scale] = prep_im_for_blob(im, conf.image_means, conf.scales, conf.max_size);
        imshow(mat2gray(im));
        hold on;
        cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'r'), ...
                   num2cell(src_rois, 2));
        cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'g'), ...
                   num2cell(target_rois, 2));
        hold off;
        %%%%%%%%%%%%%%
    end
    
    bbox_targets = sparse(bbox_targets);
end

function contained = is_contain_in_image(boxes, im_size)
    contained = boxes >= 1 & bsxfun(@le, boxes, [im_size(2), im_size(1), im_size(2), im_size(1)]);
    
    contained = all(contained, 2);
end