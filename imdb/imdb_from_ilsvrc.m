function imdb = imdb_from_ilsvrc(root_dir, image_set, year, onlyVOC20classes, flip)
% imdb = imdb_from_voc(root_dir, image_set, year)
%   Builds an image database for the PASCAL VOC devkit located
%   at root_dir using the image_set and year.
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

%imdb.name = 'ilsvrc_2014_val'
%imdb.image_dir = ''
%imdb.extension = '.JPEG'
%imdb.image_ids = {'000001', ... }
%imdb.sizes = [numimages x 2]
%imdb.classes = {'aeroplane', ... }
%imdb.num_classes
%imdb.class_to_id
%imdb.class_ids
%imdb.eval_func = pointer to the function that evaluates detections
%imdb.roidb_func = pointer to the function that returns regions of interest

if nargin < 4
    onlyVOC20classes = false;
end

if nargin < 5
    flip = false;
end

cache_file = ['./imdb/cache/imdb_ilsvrc_' year '_' image_set];
if onlyVOC20classes
    cache_file = [cache_file '_voc20c'];
end
if flip
    cache_file = [cache_file, '_flip'];
end
try
  load(cache_file);
catch
  ilsvrc_opts = get_ilsvrc_opts(root_dir, onlyVOC20classes);
  ilsvrc_opts.testset = image_set;

  imdb.image_set = image_set;
  if onlyVOC20classes
        imdb.name = ['ilsvrc_' year '_' image_set '_voc20c'];
  else
        imdb.name = ['ilsvrc_' year '_' image_set];
  end
  imdb.onlyVOC20classes = onlyVOC20classes;
  
  % remove num from image_set to get image_set_root
  num_pattern = '[^_0-9.]*';
  image_set_root = regexp(image_set, num_pattern, 'match', 'once');
  imdb.imagepath = ilsvrc_opts.imgpath.(image_set_root);
  imdb.imagedir = fileparts(ilsvrc_opts.imgpath.(image_set_root));
  if ~strcmp(image_set_root, 'test')
      imdb.annopath = ilsvrc_opts.annopath.(image_set_root);
  else
      imdb.annopath = [];
  end
  imdb.blacklist = ilsvrc_opts.blacklist.(image_set_root);
  [imdb.image_ids, ~] = textread(sprintf(ilsvrc_opts.imgsetpath, image_set), '%s %d');
  imdb.extension = 'JPEG';
  imdb.flip = flip;
  if flip
      image_at = arrayfun(@(x) sprintf('%s/%s.%s', imdb.imagedir, imdb.image_ids{x}, imdb.extension), 1:length(imdb.image_ids), 'UniformOutput', false);
      flip_image_at = arrayfun(@(x) sprintf('%s/%s_flip.%s', imdb.imagedir, imdb.image_ids{x}, imdb.extension), 1:length(imdb.image_ids), 'UniformOutput', false);
      parfor i = 1:length(imdb.image_ids)
          if ~exist(flip_image_at{i}, 'file')
             im = imread(image_at{i});
             imwrite(fliplr(im), flip_image_at{i});
          end
      end
      img_num = length(imdb.image_ids)*2;
      image_ids = imdb.image_ids;
      imdb.image_ids(1:2:img_num) = image_ids;
      imdb.image_ids(2:2:img_num) = cellfun(@(x) [x, '_flip'], image_ids, 'UniformOutput', false);
      imdb.flip_from = zeros(img_num, 1);
      imdb.flip_from(2:2:img_num) = 1:2:img_num;
  end
  imdb.classes = ilsvrc_opts.classes;
  imdb.num_classes = length(imdb.classes);
  if isfield(ilsvrc_opts, 'classes_ids2')
    imdb.classes_ids = [ilsvrc_opts.classes_ids(:); ilsvrc_opts.classes_ids2(:)];
    imdb.class_to_id = ...
        containers.Map(imdb.classes_ids, [1:imdb.num_classes, 1:imdb.num_classes]);
  else
    imdb.classes_ids = ilsvrc_opts.classes_ids(:);
    imdb.class_to_id = ...
        containers.Map(imdb.classes_ids, 1:imdb.num_classes);
  end
  imdb.class_ids = 1:imdb.num_classes;

  % private ilsvrc details
  imdb.details.ilsvrc_opts = ilsvrc_opts;

  imdb.eval_func = @imdb_eval_ilsvrc;
  imdb.roidb_func = @roidb_from_ilsvrc;
  imdb.image_at = @(i) ...
      sprintf('%s/%s.%s', imdb.imagedir, imdb.image_ids{i}, imdb.extension);

  for i = 1:length(imdb.image_ids)
    tic_toc_print('imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids));
    info = imfinfo(imdb.image_at(i));
    imdb.sizes(i, :) = [info.Height info.Width];
  end
  
%   imagedir = imdb.imagedir;
%   extension = imdb.extension;
%   image_ids = imdb.image_ids;
%   sizes = cell(length(imdb.image_ids), 1);
%   parfor i = 1:length(imdb.image_ids)
% %     tic_toc_print('imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids));
%     info = imfinfo(fullfile(imagedir, [image_ids{i} '.' extension]));
%     sizes{i} = [info.Height info.Width];
%   end
%   imdb.sizes = cell2mat(sizes);

  fprintf('Saving imdb to cache...');
  save(cache_file, 'imdb', '-v7.3');
  fprintf('done\n');
end
