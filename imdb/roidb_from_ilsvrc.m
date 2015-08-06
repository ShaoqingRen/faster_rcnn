function roidb = roidb_from_ilsvrc(imdb, varargin)
% roidb = roidb_from_voc(imdb, rootDir)
%   Builds an regions of interest database from imdb image
%   database. Uses precomputed selective search boxes available
%   in the R-CNN data package.
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addParamValue('with_hard_samples',       true,   @islogical);
ip.addParamValue('with_selective_search',   false,  @islogical);
ip.addParamValue('with_edge_box',           false,  @islogical);
ip.addParamValue('with_self_proposal',      false,  @islogical);
ip.addParamValue('rootDir',                 '.',    @ischar);
ip.addParamValue('extension',               '',     @ischar);
ip.addParamValue('compatible_with_old',     false,  @islogical);
ip.parse(imdb, varargin{:});
opts = ip.Results;

roidb.name = imdb.name;
regions_file_ss = fullfile(opts.rootDir, sprintf('/data/selective_search_data/%s%s.mat', roidb.name, opts.extension));
regions_file_eb = fullfile(opts.rootDir, sprintf('/data/edge_box_data/%s%s.mat', roidb.name, opts.extension));
regions_file_sp = fullfile(opts.rootDir, sprintf('/data/self_proposal_data/%s%s.mat', roidb.name, opts.extension));

cache_file_ss = [];
cache_file_eb = [];
cache_file_sp = [];
if opts.with_selective_search 
    cache_file_ss = 'ss_';
    if~exist(regions_file_ss, 'file')
        error('roidb_from_ilsvrc:: cannot find %s', regions_file_ss);
    end
end

if opts.with_edge_box 
    cache_file_eb = 'eb_';
    if ~exist(regions_file_eb, 'file')
        error('roidb_from_ilsvrc:: cannot find %s', regions_file_eb);
    end
end

if opts.with_self_proposal 
    cache_file_sp = 'sp_';
    if ~exist(regions_file_sp, 'file')
        error('roidb_from_ilsvrc:: cannot find %s', regions_file_sp);
    end
end

cache_file = fullfile(opts.rootDir, ['/imdb/cache/roidb_' cache_file_ss cache_file_eb cache_file_sp imdb.name opts.extension]);
if imdb.flip
    cache_file = [cache_file '_flip'];
end
try
  load(cache_file);
catch
  ilsvrc_opts = imdb.details.ilsvrc_opts;

  addpath(fullfile(ilsvrc_opts.rootdir, 'ILSVRC2014_devkit\evaluation')); 
  
  try
      fprintf('Loading region proposals...');
      regions = [];
      if opts.with_selective_search
            regions = load_proposals(regions_file_ss, regions);
      end
      if opts.with_edge_box
            regions = load_proposals(regions_file_eb, regions);
      end
      if opts.with_self_proposal
            regions = load_proposals(regions_file_sp, regions);
      end
      fprintf('done\n');
  catch
      error('roidb_from_ilsvrc:: cannot load %s or %s or %s', regions_file_ss, regions_file_eb, regions_file_sp);
      fprintf('Failed to Load %s.\n', sprintf('/data/selective_search_data/%s.mat', roidb.name));
%       ipt = input('Go on without selective data ? Yes ? ', 's');
%       if ipt ~= 'y'
%       error('Failed to Load selective_search_data');
%       end
  end
  if isempty(regions)
      fprintf('Warrning: no windows proposal is loaded !\n');
      regions.boxes = cell(length(imdb.image_ids), 1);
      if imdb.flip
            regions.images = imdb.image_ids(1:2:end);
      else
            regions.images = imdb.image_ids;
      end
  end

  % remove num from image_set to get image_set_root
  num_pattern = '[^_0-9.]*(?=[_0-9.]*)';
  image_set_root = regexp(imdb.image_set, num_pattern, 'match', 'once');
  
  if ~imdb.flip
      rois = cell(length(imdb.image_ids), 1);
      for i = 1:length(imdb.image_ids)
%       parfor i = 1:length(imdb.image_ids)
        tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids));
        if strcmp(image_set_root, 'test')
            voc_rec = [];
        else
            try
                voc_rec = VOCreadrecxml(sprintf(ilsvrc_opts.annopath.(image_set_root), imdb.image_ids{i}), ilsvrc_opts.hash);
            catch
                fprintf('cannot find %s.\n', sprintf(ilsvrc_opts.annopath.(image_set_root), imdb.image_ids{i}));
                voc_rec = [];
            end
        end
        if ~isempty(regions)
            [~, image_name1] = fileparts(imdb.image_ids{i});
            [~, image_name2] = fileparts(regions.images{i});
            assert(strcmp(image_name1, image_name2));        
        end
        rois{i} = attach_proposals(voc_rec, regions.boxes{i}, imdb.class_to_id, imdb.classes, imdb.onlyVOC20classes, false);
      end
      roidb.rois = cat(1, rois{:});
  else
      rois_ = cell(length(imdb.image_ids)/2, 1);
      rois_flip = cell(length(imdb.image_ids)/2, 1);
      for i = 1:length(imdb.image_ids)/2
%       parfor i = 1:length(imdb.image_ids)/2
        tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i*2, length(imdb.image_ids));
        try
            voc_rec = VOCreadrecxml(sprintf(ilsvrc_opts.annopath.(image_set_root), imdb.image_ids{i*2-1}), ilsvrc_opts.hash);
        catch
            fprintf('cannot find %s.\n', sprintf(ilsvrc_opts.annopath.(image_set_root), imdb.image_ids{i*2-1}));
            voc_rec = [];
        end
        if ~isempty(regions)
            [~, image_name1] = fileparts(imdb.image_ids{i*2-1});
            [~, image_name2] = fileparts(regions.images{i});
            assert(strcmp(image_name1, image_name2));
            assert(imdb.flip_from(i*2) == i*2-1);
        end
        rois_{i} = attach_proposals(voc_rec, regions.boxes{i}, imdb.class_to_id, imdb.classes, imdb.onlyVOC20classes, false);
        rois_flip{i} = attach_proposals(voc_rec, regions.boxes{i}, imdb.class_to_id, imdb.classes, imdb.onlyVOC20classes, true);
      end
      rois = cell(length(imdb.image_ids), 1);
      rois(1:2:end) = rois_;
      rois(2:2:end) = rois_flip;
      roidb.rois = cat(1, rois{:});
  end
  
  fprintf('Saving roidb to cache...');
  save(cache_file, 'roidb', '-v7.3');
  fprintf('done\n');
end


% ------------------------------------------------------------------------
function rec = attach_proposals(voc_rec, boxes, class_to_id, class, onlyVOC20classes, flip)
% ------------------------------------------------------------------------

% change selective search order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
if ~isempty(boxes)
    boxes = boxes(:, [2 1 4 3]);
end
if flip && ~isempty(boxes)
    boxes(:, [1, 3]) = voc_rec.imgsize(1) + 1 - boxes(:, [3, 1]);
end

%           gt: [2108x1 double]
%      overlap: [2108x20 single]
%      dataset: 'voc_2007_trainval'
%        boxes: [2108x4 single]
%         feat: [2108x9216 single]
%        class: [2108x1 uint8]
if isfield(voc_rec, 'objects') && ~isempty(voc_rec.('objects'))
  if ~onlyVOC20classes
      valid_objects = true(1, length(voc_rec.objects(:)));
  else
      valid_objects = true(1, length(voc_rec.objects(:))) & class_to_id.isKey({voc_rec.objects(:).class});
  end
  gt_boxes = cat(1, voc_rec.objects(valid_objects).bbox) + 1; %trans to matlab index
  if flip && ~isempty(gt_boxes)
    gt_boxes(:, [1, 3]) = voc_rec.imgsize(1) + 1 - gt_boxes(:, [3, 1]);
  end
  all_boxes = cat(1, gt_boxes, boxes);
  gt_classes = class_to_id.values({voc_rec.objects(valid_objects).class});
  gt_classes = cat(1, gt_classes{:});
  num_gt_boxes = size(gt_boxes, 1);
else
  gt_boxes = nan(0, 4);
  all_boxes = boxes;
  gt_classes = [];
  num_gt_boxes = 0;
end
num_boxes = size(boxes, 1);

rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));
rec.overlap = sparse(zeros(num_gt_boxes+num_boxes, class_to_id.Count));
for i = 1:num_gt_boxes
  rec.overlap(:, gt_classes(i)) = ...
      max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
end
rec.boxes = single(all_boxes);
if isempty(rec.boxes)
    rec.boxes = zeros(0, 4);
end
rec.feat = [];
rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));

% ------------------------------------------------------------------------
function regions = load_proposals(proposal_file, regions)
% ------------------------------------------------------------------------
if isempty(regions)
    regions = load(proposal_file);
else
    regions_more = load(proposal_file);
    if ~all(cellfun(@(x, y) strcmp(x, y), regions.images(:), regions_more.images(:), 'UniformOutput', true))
        error('roidb_from_ilsvrc: %s is has different images list with other proposals.\n', proposal_file);
    end
    regions.boxes = cellfun(@(x, y) [double(x); double(y)], regions.boxes, regions_more.boxes, 'UniformOutput', false);
end
