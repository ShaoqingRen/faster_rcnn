function conf = proposal_config(varargin)
% conf = proposal_config(varargin)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    ip = inputParser;
    
    %% training
    ip.addParamValue('use_gpu',         gpuDeviceCount > 0, ...            
                                                        @islogical);
                                    
    % whether drop the anchors that has edges outside of the image boundary
    ip.addParamValue('drop_boxes_runoff_image', ...
                                        true,           @islogical);
    
    % Image scales -- the short edge of input image                                                                                                
    ip.addParamValue('scales',          600,            @ismatrix);
    % Max pixel size of a scaled input image
    ip.addParamValue('max_size',        1000,           @isscalar);
    % Images per batch, only supports ims_per_batch = 1 currently
    ip.addParamValue('ims_per_batch',   1,              @isscalar);
    % Minibatch size
    ip.addParamValue('batch_size',      256,            @isscalar);
    % Fraction of minibatch that is foreground labeled (class > 0)
    ip.addParamValue('fg_fraction',     0.5,           @isscalar);
    % weight of background samples, when weight of foreground samples is
    % 1.0
    ip.addParamValue('bg_weight',       1.0,            @isscalar);
    % Overlap threshold for a ROI to be considered foreground (if >= fg_thresh)
    ip.addParamValue('fg_thresh',       0.7,            @isscalar);
    % Overlap threshold for a ROI to be considered background (class = 0 if
    % overlap in [bg_thresh_lo, bg_thresh_hi))
    ip.addParamValue('bg_thresh_hi',    0.3,            @isscalar);
    ip.addParamValue('bg_thresh_lo',    0,              @isscalar);
    % mean image, in RGB order
    ip.addParamValue('image_means',     128,            @ismatrix);
    % Use horizontally-flipped images during training?
    ip.addParamValue('use_flipped',     true,           @islogical);
    % Stride in input image pixels at ROI pooling level (network specific)
    % 16 is true for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
    ip.addParamValue('feat_stride',     16,             @isscalar);
    % train proposal target only to labled ground-truths or also include
    % other proposal results (selective search, etc.)
    ip.addParamValue('target_only_gt',  true,           @islogical);

    % random seed                    
    ip.addParamValue('rng_seed',        6,              @isscalar);

    
    %% testing
    ip.addParamValue('test_scales',     600,            @isscalar);
    ip.addParamValue('test_max_size',   1000,           @isscalar);
    ip.addParamValue('test_nms',        0.3,            @isscalar);
    ip.addParamValue('test_binary',     false,          @islogical);
    ip.addParamValue('test_min_box_size',16,            @isscalar);
    ip.addParamValue('test_drop_boxes_runoff_image', ...
                                        false,          @islogical);
    
    ip.parse(varargin{:});
    conf = ip.Results;
    
    assert(conf.ims_per_batch == 1, 'currently rpn only supports ims_per_batch == 1');
    
    % if image_means is a file, load it
    if ischar(conf.image_means)
        s = load(conf.image_means);
        s_fieldnames = fieldnames(s);
        assert(length(s_fieldnames) == 1);
        conf.image_means = s.(s_fieldnames{1});
    end
end