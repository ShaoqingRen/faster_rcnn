function [imdbs, roidbs] = subsample_images(imdbs, roidbs, max_num_neg_images, seed)

if ~exist('seed', 'var')
  seed = 6;
end

% class_num = cellfun(@(x) length(x.class_ids), imdbs, 'UniformOutput', true);
% assert(length(unique(class_num)) == 1);
% class_num = unique(class_num);

rois = cellfun(@(x) x.rois(:), roidbs, 'UniformOutput', false);
rois_combine = cell2mat(rois(:));

% fix the random seed for repeatability
prev_rng = seed_rand(seed);
inds = randperm(length(rois_combine), max_num_neg_images);
inds = sort(inds);

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

% restore previous rng
rng(prev_rng);

end