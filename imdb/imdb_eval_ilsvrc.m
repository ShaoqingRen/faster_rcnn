function res = imdb_eval_ilsvrc(boxes, imdb, cache_name, suffix, fast)
% res = imdb_eval_voc(cls, boxes, imdb, suffix)
%   Use the VOCdevkit to evaluate detections specified in boxes
%   for class cls against the ground-truth boxes in the image
%   database imdb. Results files are saved with an optional
%   suffix.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% Add a random string ("salt") to the end of the results file name
% to prevent concurrent evaluations from clobbering each other
use_res_salt = true;
% Delete results files after computing APs
rm_res = true;
% comp4 because we use outside data (ILSVRC2012)
comp_id = 'ilsvrc2014';
% draw each class curve
draw_curve = true;
display_progress = true;
if fast
    draw_curve = false;
    display_progress = false;
end

% save results
if ~exist('suffix', 'var') || isempty(suffix) || strcmp(suffix, '')
  suffix = '';
else
  if suffix(1) ~= '_'
    suffix = ['_' suffix];
  end
end

conf.cache_dir = fullfile('output', 'fast_rcnn_cachedir', cache_name, imdb.name);
ilsvrc_opts  = imdb.details.ilsvrc_opts;
image_ids = imdb.image_ids;
test_set = ilsvrc_opts.testset;
year = ilsvrc_opts.dataset(4:end);

addpath(fullfile(ilsvrc_opts.rootdir, 'ILSVRC2014_devkit\evaluation')); 

if use_res_salt
  timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
  res_id = [comp_id '_' timestamp];
else
  res_id = comp_id;
end
res_fn = fullfile(ilsvrc_opts.resdir, res_id);

boxes_t = cell(length(image_ids), 1);
for i = 1:length(image_ids)
    boxes_t{i} = cell(length(boxes), 1);
    for c = 1:length(boxes)
        boxes_t{i}{c} = boxes{c}{i};
    end
end

if ~strcmp(test_set, 'test') && fast
    % to test faster, for not test set, we don't write file here

    predict_rst = cell(length(image_ids), 1);
    nBoxes = length(boxes);
    parfor i = 1:length(image_ids);
%     for i = 1:length(image_ids);
        img_id = i;
        predict_rst_sub = cell(nBoxes, 1);
        for c = 1:nBoxes
              bbox = boxes_t{i}{c};
              keep = nms(bbox, 0.3);
              bbox = bbox(keep,:);
              class_id = ilsvrc_opts.valid_classes(c);

              idx = ones(size(bbox, 1), 1);
              predict_rst_sub{c} = [img_id(idx), class_id(idx), bbox(: ,end), bbox(:, 1:4)];
        end
        predict_rst{i} = cell2mat(predict_rst_sub);
    end 
    predict_rst = cell2mat(predict_rst);
    predict_file = predict_rst;

    tic;
    gtruth_dir = fileparts(imdb.annopath);
    meta_file = ilsvrc_opts.meta_data;
    eval_file = sprintf(ilsvrc_opts.imgsetpath, imdb.image_set);
    blacklist_file = imdb.blacklist;
    optional_cache_file = fullfile(ilsvrc_opts.localdir, imdb.image_set);
    [ap, recall, prec] = eval_detection(predict_file,gtruth_dir,meta_file,...
            eval_file,blacklist_file,optional_cache_file);
    ap_auc = cellfun(@(x, y) xVOCap(x, y), recall, prec, 'UniformOutput', true);

    for c = 1:length(ilsvrc_opts.valid_classes)
        class_id = ilsvrc_opts.valid_classes(c);
        fprintf('!!! %s : %.4f %.4f\n', imdb.classes{c}, ap(class_id), ap_auc(class_id));
    end

    save(fullfile(conf.cache_dir, ['pr_' imdb.name suffix]), ...
        'recall', 'prec', 'ap', 'ap_auc');

    res.recall = recall(ilsvrc_opts.valid_classes);
    res.prec = prec(ilsvrc_opts.valid_classes);
    res.ap = ap(ilsvrc_opts.valid_classes);
    res.ap_auc = ap_auc(ilsvrc_opts.valid_classes);
    
else
    % write out detections in ilsvrc format and score
    th = tic();
    fprintf('Writing result file...');
    fid = fopen(res_fn, 'w');
    for i = 1:length(image_ids);
        for c = 1:length(boxes)
              bbox = boxes{c}{i};
              keep = nms(bbox, 0.3);
              bbox = bbox(keep,:);
              class_id = ilsvrc_opts.valid_classes(c);
              for j = 1:size(bbox,1)
                  %[img_ids obj_labels obj_confs xmin ymin xmax ymax]
                fprintf(fid, '%d %d %.4f %.0f %.0f %.0f %.0f\n', i, class_id, bbox(j,end), bbox(j,1:4));
              end
        end
    end
    fclose(fid);
    fprintf('Done in %fs\n', toc(th));
    
    if ~strcmp(test_set, 'test')
        tic;
        predict_file = res_fn;
        gtruth_dir = fileparts(imdb.annopath);
        meta_file = ilsvrc_opts.meta_data;
        eval_file = sprintf(ilsvrc_opts.imgsetpath, imdb.image_set);
        blacklist_file = imdb.blacklist;
        optional_cache_file = fullfile(ilsvrc_opts.localdir, imdb.image_set);
        [ap, recall, prec] = eval_detection(predict_file,gtruth_dir,meta_file,...
                eval_file,blacklist_file,optional_cache_file);
        ap_auc = cellfun(@(x, y) xVOCap(x, y), recall, prec, 'UniformOutput', true);

        for c = 1:length(ilsvrc_opts.valid_classes)
            class_id = ilsvrc_opts.valid_classes(c);
            fprintf('!!! %s : %.4f %.4f\n', imdb.classes{c}, ap(class_id), ap_auc(class_id));
        end

        save(fullfile(conf.cache_dir, ['pr_' imdb.name suffix]), ...
            'recall', 'prec', 'ap', 'ap_auc');

        res.recall = recall(ilsvrc_opts.valid_classes);
        res.prec = prec(ilsvrc_opts.valid_classes);
        res.ap = ap(ilsvrc_opts.valid_classes);
        res.ap_auc = ap_auc(ilsvrc_opts.valid_classes);
    else
        res = [];
    end
end

if rm_res
  if exist(res_fn, 'file')
      delete(res_fn);
  end
end

rmpath(fullfile(ilsvrc_opts.rootdir, 'ILSVRC2014_devkit\evaluation')); 

