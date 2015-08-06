function startup()
% startup()
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    curdir = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(curdir, 'utils')));
    addpath(genpath(fullfile(curdir, 'functions')));
    addpath(genpath(fullfile(curdir, 'bin')));
    addpath(genpath(fullfile(curdir, 'experiments')));

    data_folder = fullfile(fileparts(pwd), 'faster_rcnn_data');
    link_data(data_folder, 'data');
    link_data(data_folder, 'datasets');
    link_data(data_folder, 'external', true);
    addpath(genpath(fullfile(pwd, 'imdb')));
    link_data(data_folder, fullfile('imdb', 'cache'));

    data_folder = fullfile(fileparts(pwd), 'faster_rcnn_data');
    link_data(data_folder, 'output');
    link_data(data_folder, 'models');
    fprintf('fast_rcnn startup done\n');
    
end

function link_data(data_folder, sub_dir, isaddpath)    
    if ~exist('isaddpath', 'var')
        isaddpath = false;
    end

    if ~exist(fullfile(data_folder, sub_dir), 'dir')
       fprintf('miss target folder %s\n', fullfile(data_folder, sub_dir));
       return;
    end
    if ~exist(fullfile(pwd, sub_dir), 'dir')
        symbolic_link(fullfile(pwd, sub_dir), fullfile(data_folder, sub_dir));
    end
    if isaddpath
        addpath(genpath(fullfile(pwd, sub_dir)));
    end
end
