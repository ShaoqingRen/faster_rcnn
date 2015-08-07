function script_fast_rcnn_VOC2007_ZF()
% script_fast_rcnn_VOC2007_ZF()
% Fast rcnn training and testing with Zeiler & Fergus model
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% model
model                       = Model.ZF_for_Fast_RCNN_VOC2007();
% cache name
opts.cache_name             = 'fast_rcnn_VOC2007_ZF';
% config
conf                        = fast_rcnn_config('image_means', model.mean_image);
% train/test data
dataset                     = [];
dataset                     = Dataset.voc2007_trainval_ss(dataset, 'train', conf.use_flipped);
dataset                     = Dataset.voc2007_test_ss(dataset, 'test', false);

% do validation, or not
opts.do_val                 = true; 

%% -------------------- TRAINING --------------------

opts.fast_rcnn_model        = fast_rcnn_train(conf, dataset.imdb_train, dataset.roidb_train, ...
                                'do_val',           opts.do_val, ...
                                'imdb_val',         dataset.imdb_test, ...
                                'roidb_val',        dataset.roidb_test, ...
                                'solver_def_file',  model.solver_def_file, ...
                                'net_file',         model.net_file, ...
                                'cache_name',       opts.cache_name);
assert(exist(opts.fast_rcnn_model, 'file') ~= 0, 'not found trained model');

                                
%% -------------------- TESTING --------------------
                              fast_rcnn_test(conf, dataset.imdb_test, dataset.roidb_test, ...
                                    'net_def_file',     model.test_net_def_file, ...
                                    'net_file',         opts.fast_rcnn_model, ...
                                    'cache_name',       opts.cache_name);

                                
end
