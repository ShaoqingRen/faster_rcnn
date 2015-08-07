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
    
    mkdir_if_missing(fullfile(curdir, 'data'));
    addpath(genpath(fullfile(curdir, 'data')));
    
    mkdir_if_missing(fullfile(curdir, 'datasets'));
    addpath(genpath(fullfile(curdir, 'datasets')));
    
    mkdir_if_missing(fullfile(curdir, 'external'));
    addpath(genpath(fullfile(curdir, 'external')));
    
    mkdir_if_missing(fullfile(curdir, 'imdb', 'cache'));
    addpath(genpath(fullfile(curdir, 'imdb', 'cache')));
    
    mkdir_if_missing(fullfile(curdir, 'output'));
    addpath(genpath(fullfile(curdir, 'output')));
    
    mkdir_if_missing(fullfile(curdir, 'models'));
    addpath(genpath(fullfile(curdir, 'models')));

    fprintf('fast_rcnn startup done\n');
end
