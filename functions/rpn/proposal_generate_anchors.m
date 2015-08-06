function anchors = proposal_generate_anchors(cache_name, varargin)
% anchors = proposal_generate_anchors(cache_name, varargin)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs
    ip = inputParser;
    ip.addRequired('cache_name',                        @isstr);

    % the size of the base anchor 
    ip.addParamValue('base_size',       16,             @isscalar);
    % ratio list of anchors
    ip.addParamValue('ratios',          [0.5, 1, 2],    @ismatrix);
    % scale list of anchors
    ip.addParamValue('scales',          2.^[3:5],       @ismatrix);    
    ip.addParamValue('ignore_cache',    false,          @islogical);
    ip.parse(cache_name, varargin{:});
    opts = ip.Results;

%%
    if ~opts.ignore_cache
        anchor_cache_dir            = fullfile(pwd, 'output', 'rpn_cachedir', cache_name); 
                                      mkdir_if_missing(anchor_cache_dir);
        anchor_cache_file           = fullfile(anchor_cache_dir, 'anchors');
    end
    try
        ld                      = load(anchor_cache_file);
        anchors                 = ld.anchors;
    catch
        base_anchor             = [1, 1, opts.base_size, opts.base_size];
        ratio_anchors           = ratio_jitter(base_anchor, opts.ratios);
        anchors                 = cellfun(@(x) scale_jitter(x, opts.scales), num2cell(ratio_anchors, 2), 'UniformOutput', false);
        anchors                 = cat(1, anchors{:});
        if ~opts.ignore_cache
            save(anchor_cache_file, 'anchors');
        end
    end
    
end

function anchors = ratio_jitter(anchor, ratios)
    ratios = ratios(:);
    
    w = anchor(3) - anchor(1) + 1;
    h = anchor(4) - anchor(2) + 1;
    x_ctr = anchor(1) + (w - 1) / 2;
    y_ctr = anchor(2) + (h - 1) / 2;
    size = w * h;
    
    size_ratios = size ./ ratios;
    ws = round(sqrt(size_ratios));
    hs = round(ws .* ratios);
    
    anchors = [x_ctr - (ws - 1) / 2, y_ctr - (hs - 1) / 2, x_ctr + (ws - 1) / 2, y_ctr + (hs - 1) / 2];
end

function anchors = scale_jitter(anchor, scales)
    scales = scales(:);

    w = anchor(3) - anchor(1) + 1;
    h = anchor(4) - anchor(2) + 1;
    x_ctr = anchor(1) + (w - 1) / 2;
    y_ctr = anchor(2) + (h - 1) / 2;

    ws = w * scales;
    hs = h * scales;
    
    anchors = [x_ctr - (ws - 1) / 2, y_ctr - (hs - 1) / 2, x_ctr + (ws - 1) / 2, y_ctr + (hs - 1) / 2];
end

