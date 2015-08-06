function [imdbs, roidbs] = subsample_images_per_class(imdbs, roidbs, max_per_class_image_num, seed)

if ~exist('seed', 'var')
  seed = 6;
end

class_num = cellfun(@(x) length(x.class_ids), imdbs, 'UniformOutput', true);
assert(length(unique(class_num)) == 1);
class_num = unique(class_num);

rois = cellfun(@(x) x.rois, roidbs, 'UniformOutput', false);
rois_combine = cell2mat(rois(:));
rois_combine_class = arrayfun(@(x) x.class, rois_combine, 'UniformOutput', false);

%% select images with max_image_num

% fix the random seed for repeatability
prev_rng = seed_rand(seed);
inds = cell(class_num, 1);
rois_combine_length = length(rois_combine);
valid_idxs = cell(class_num, 1);
parfor i = 1:class_num
    valid_idxs{i} = cellfun(@(x) any(x == i), rois_combine_class, 'UniformOutput', false);
    valid_idxs{i} = cell2mat(valid_idxs{i});
end

for i = 1:class_num
    valid_num = sum(valid_idxs{i});

    num = min(valid_num, max_per_class_image_num);
    inds{i} = 1:rois_combine_length;
    inds{i} = inds{i}(valid_idxs{i});
    inds{i} = inds{i}(randperm(length(inds{i}), num));
end

inds = cell2mat(inds')';
inds = unique(inds);

% restore previous rng
rng(prev_rng);

img_idx_start = 1;
for i = 1:length(imdbs)
    imdb_img_num = length(imdbs{i}.image_ids);
    img_idx_end = img_idx_start + imdb_img_num - 1;
    inds_start = find(inds >= img_idx_start, 1, 'first');
    inds_end = find(inds <= img_idx_end, 1, 'last');

    inds_sub = inds(inds_start:inds_end);
    inds_sub = inds_sub - img_idx_start + 1;

    imdbs{i}.image_ids = imdbs{i}.image_ids(inds_sub);
    imdbs{i}.sizes = imdbs{i}.sizes(inds_sub, :);
    if isfield(imdbs{i}, 'image_dir')
        imdbs{i}.image_at = @(x) ...
          sprintf('%s/%s.%s', imdbs{i}.image_dir, imdbs{i}.image_ids{x}, imdbs{i}.extension);
    else
        imdbs{i}.image_at = @(x) ...
          sprintf('%s/%s.%s', imdbs{i}.imagedir, imdbs{i}.image_ids{x}, imdbs{i}.extension);
    end
    roidbs{i}.rois = roidbs{i}.rois(inds_sub);

    img_idx_start = img_idx_start + imdb_img_num;
end


