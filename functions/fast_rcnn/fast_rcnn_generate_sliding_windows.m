function roidb = fast_rcnn_generate_sliding_windows(conf, imdb, roidb, roipool_in_size)
% [pred_boxes, scores] = fast_rcnn_conv_feat_detect(conf, im, conv_feat, boxes, max_rois_num_in_gpu, net_idx)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    regions.images = imdb.image_ids;
    
    im_sizes = imdb.sizes;
    regions.boxes = cellfun(@(x) generate_sliding_windows_one_image(conf, x, roipool_in_size), num2cell(im_sizes, 2), 'UniformOutput', false);

    roidb = roidb_from_proposal(imdb, roidb, regions);
end

function boxes = generate_sliding_windows_one_image(conf, im_size, roipool_in_size)
    im_scale = prep_im_for_blob_size(im_size, conf.scales, conf.max_size);
    im_size = round(im_size * im_scale);

    x1 = 1:conf.feat_stride:im_size(2);
    y1 = 1:conf.feat_stride:im_size(1);
    [x1, y1] = meshgrid(x1, y1);
    x1 = x1(:);
    y1 = y1(:);
    x2 = x1 + roipool_in_size * conf.feat_stride - 1;
    y2 = y1 + roipool_in_size * conf.feat_stride - 1;
    
    boxes = [x1, y1, x2, y2];
    boxes = filter_boxes(im_size, boxes);
    
    boxes = bsxfun(@times, boxes-1, 1/im_scale) + 1;
end

function boxes = filter_boxes(im_size, boxes)    
    valid_ind = boxes(:, 1) >= 1 & boxes(:, 1) <= im_size(2) & ...
                boxes(:, 2) >= 1 & boxes(:, 2) <= im_size(1) & ...
                boxes(:, 3) >= 1 & boxes(:, 3) <= im_size(2) & ...
                boxes(:, 4) >= 1 & boxes(:, 4) <= im_size(1);

    boxes = boxes(valid_ind, :);
end