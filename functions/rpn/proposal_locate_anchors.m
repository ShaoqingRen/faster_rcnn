function [anchors, im_scales] = proposal_locate_anchors(conf, im_size, target_scale, feature_map_size)
% [anchors, im_scales] = proposal_locate_anchors(conf, im_size, target_scale, feature_map_size)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   
% generate anchors for each scale

    % only for fcn
    if ~exist('feature_map_size', 'var')
        feature_map_size = [];
    end

    func = @proposal_locate_anchors_single_scale;

    if exist('target_scale', 'var')
        [anchors, im_scales] = func(im_size, conf, target_scale, feature_map_size);
    else
        [anchors, im_scales] = arrayfun(@(x) func(im_size, conf, x, feature_map_size), ...
            conf.scales, 'UniformOutput', false);
    end

end

function [anchors, im_scale] = proposal_locate_anchors_single_scale(im_size, conf, target_scale, feature_map_size)
    if isempty(feature_map_size)
        im_scale = prep_im_for_blob_size(im_size, target_scale, conf.max_size);
        img_size = round(im_size * im_scale);
        output_size = cell2mat([conf.output_height_map.values({img_size(1)}), conf.output_width_map.values({img_size(2)})]);
    else
        im_scale = prep_im_for_blob_size(im_size, target_scale, conf.max_size);
        output_size = feature_map_size;
    end
    
    shift_x = [0:(output_size(2)-1)] * conf.feat_stride;
    shift_y = [0:(output_size(1)-1)] * conf.feat_stride;
    [shift_x, shift_y] = meshgrid(shift_x, shift_y);
    
    % concat anchors as [channel, height, width], where channel is the fastest dimension.
    anchors = reshape(bsxfun(@plus, permute(conf.anchors, [1, 3, 2]), ...
        permute([shift_x(:), shift_y(:), shift_x(:), shift_y(:)], [3, 1, 2])), [], 4);
    
%   equals to  
%     anchors = arrayfun(@(x, y) single(bsxfun(@plus, conf.anchors, [x, y, x, y])), shift_x, shift_y, 'UniformOutput', false);
%     anchors = reshape(anchors, [], 1);
%     anchors = cat(1, anchors{:});

end