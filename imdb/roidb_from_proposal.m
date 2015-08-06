function roidb = roidb_from_proposal(imdb, roidb, regions, varargin)
% roidb = roidb_from_proposal(imdb, roidb, regions, varargin)s
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addRequired('roidb', @isstruct);
ip.addRequired('regions', @isstruct);
ip.addParamValue('keep_raw_proposal', true, @islogical);
ip.parse(imdb, roidb, regions, varargin{:});
opts = ip.Results;

assert(strcmp(opts.roidb.name, opts.imdb.name));
rois = opts.roidb.rois;

if ~opts.keep_raw_proposal
    % remove proposal boxes in roidb
    for i = 1:length(rois)  
        is_gt = rois(i).gt;
        rois(i).gt = rois(i).gt(is_gt, :);
        rois(i).overlap = rois(i).overlap(is_gt, :);
        rois(i).boxes = rois(i).boxes(is_gt, :);
        rois(i).class = rois(i).class(is_gt, :);
    end
end

% add new proposal boxes
for i = 1:length(rois)  
    [~, image_name1] = fileparts(imdb.image_ids{i});
    [~, image_name2] = fileparts(opts.regions.images{i});
    assert(strcmp(image_name1, image_name2));
    
    boxes = opts.regions.boxes{i}(:, 1:4);
    is_gt = rois(i).gt;
    gt_boxes = rois(i).boxes(is_gt, :);
    gt_classes = rois(i).class(is_gt, :);
    all_boxes = cat(1, rois(i).boxes, boxes);
    
    num_gt_boxes = size(gt_boxes, 1);
    num_boxes = size(boxes, 1);
    
    rois(i).gt = cat(1, rois(i).gt, false(num_boxes, 1));
    rois(i).overlap = cat(1, rois(i).overlap, zeros(num_boxes, size(rois(i).overlap, 2)));
    rois(i).boxes = cat(1, rois(i).boxes, boxes);
    rois(i).class = cat(1, rois(i).class, zeros(num_boxes, 1));
    for j = 1:num_gt_boxes
        rois(i).overlap(:, gt_classes(j)) = ...
            max(full(rois(i).overlap(:, gt_classes(j))), boxoverlap(all_boxes, gt_boxes(j, :))); 
    end
end

roidb.rois = rois;

end