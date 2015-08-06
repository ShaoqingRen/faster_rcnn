function [centroid_idx, centroids] = proposal_label_cluster(method, imdbs, roidbs, centroid_number)
% [centroid_idx, centroids] = proposal_label_cluster(method, imdbs, roidbs, centroid_number)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   

if ~iscell(imdbs)
    imdbs = {imdbs};
    roidbs = {roidbs};
end

imdbs = imdbs(:);
roidbs = roidbs(:);

im_sizes = cellfun(@(x) x.sizes, imdbs, 'UniformOutput', false);
im_sizes = cell2mat(im_sizes);
im_sizes = num2cell(im_sizes, 2);

rois = cellfun(@(x) x.rois(:), roidbs, 'UniformOutput', false);
rois_combine = cell2mat(rois(:));
boxes = {rois_combine.boxes};
boxes = boxes(:);

is_gt = {rois_combine.gt};
is_gt = is_gt(:);
transformed_boxes = cellfun(@norm_img, boxes, im_sizes, 'UniformOutput', false);
transformed_boxes = cellfun(@(x) proposal_kmeans_box_transform(method, x), transformed_boxes, 'UniformOutput', false);
transformed_boxes = cellfun(@(x, y) x(y, :), transformed_boxes, is_gt, 'UniformOutput', false);

[centroid_idx, centroids] = kmeans(cell2mat(transformed_boxes), centroid_number);

centroids = proposal_kmeans_box_transform_inv(method, centroids);

debug = 0;
if debug
    gt_boxes = cellfun(@(x, y) x(y, :), boxes, is_gt, 'UniformOutput', false);
    ss_boxes = cellfun(@(x, y) x(~y, :), boxes, is_gt, 'UniformOutput', false);
    thrs = 0.5:0.05:1;

    perRectRecall_ss = DetectionRecall(ss_boxes, gt_boxes, thrs);
    ss_boxes = cellfun(@(x) x(1:min(centroid_number, size(x, 1)), :), ss_boxes, 'UniformOutput', false);
    perRectRecall_ss_sub = DetectionRecall(ss_boxes, gt_boxes, thrs);
    test_boxes = cellfun(@(x) norm_img_inv(centroids, x), im_sizes, 'UniformOutput', false);
    perRectRecall_center = DetectionRecall(test_boxes, gt_boxes, thrs);
    plot(thrs, perRectRecall_ss, thrs, perRectRecall_ss_sub, thrs, perRectRecall_center, 'LineWidth', 2);
    
    legend('selective search', sprintf('selective search %d', centroid_number), 'kmean');
end

end

function transformed_box = norm_img(box, im_size)
    if isempty(box)
        box = nan(0, 4);
    end
    transformed_box = bsxfun(@rdivide, double(box), [im_size(2) im_size(1) im_size(2) im_size(1)]);
end

function transformed_box = norm_img_inv(box, im_size)
    if isempty(box)
        box = nan(0, 4);
    end
    transformed_box = bsxfun(@times, double(box), [im_size(2) im_size(1) im_size(2) im_size(1)]);
end

function vis_boxes(boxes)
    sz = 600;
    boxes = boxes * sz;
    imshow(ones(sz, sz, 3));
    cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x)), num2cell(boxes, 2));
end