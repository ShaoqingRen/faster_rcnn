function gather_rpn_fast_rcnn_models(conf_proposal, conf_fast_rcnn, model, dataset)
    cachedir = fullfile(pwd, 'output', 'rpn_fast_rcnn', model.final_model.cache_name);
    mkdir_if_missing(cachedir);
    
    % find latest model for rpn and fast rcnn
    [rpn_test_net_def_file, rpn_output_model_file] = find_last_output_model_file(model.stage1_rpn, model.stage2_rpn);
    [fast_rcnn_test_net_def_file, fast_rcnn_output_model_file] = find_last_output_model_file(model.stage1_fast_rcnn, model.stage2_fast_rcnn);
    
    % check whether feature shared and find the indexs of shared layers
    [is_share_feature, ~, fast_rcnn_weights, shared_layer_idx] = ...
        check_proposal_fast_rcnn_model(rpn_test_net_def_file, rpn_output_model_file, ...
         fast_rcnn_test_net_def_file, fast_rcnn_output_model_file);
     
    proposal_detection_model.classes = dataset.imdb_test.classes;
    proposal_detection_model.image_means = conf_proposal.image_means;
    proposal_detection_model.conf_proposal = conf_proposal;
    proposal_detection_model.conf_detection = conf_fast_rcnn;
    
    % copy rpn and fast rcnn models into cachedir
    [~, test_net_proposal_name, test_net_proposal_ext] = fileparts(rpn_test_net_def_file);
    proposal_detection_model.proposal_net_def = ['proposal_', test_net_proposal_name, test_net_proposal_ext];
    [~, proposal_model_name, proposal_model_ext] = fileparts(rpn_output_model_file);
    proposal_detection_model.proposal_net = ['proposal_', proposal_model_name, proposal_model_ext];
    [~, test_net_fast_rcnn_name, test_net_fast_rcnn_ext] = fileparts(fast_rcnn_test_net_def_file);
    proposal_detection_model.detection_net_def = ['detection_', test_net_fast_rcnn_name, test_net_fast_rcnn_ext];
    [~, fast_rcnn_model_name, fast_rcnn_model_ext] = fileparts(fast_rcnn_output_model_file);
    proposal_detection_model.detection_net = ['detection_', fast_rcnn_model_name, fast_rcnn_model_ext];
     
    copyfile(rpn_test_net_def_file, fullfile(cachedir, proposal_detection_model.proposal_net_def));
    copyfile(rpn_output_model_file, fullfile(cachedir, proposal_detection_model.proposal_net));
    copyfile(fast_rcnn_test_net_def_file, fullfile(cachedir, proposal_detection_model.detection_net_def));
    copyfile(fast_rcnn_output_model_file, fullfile(cachedir, proposal_detection_model.detection_net));
    
    proposal_detection_model.is_share_feature = is_share_feature;
    if is_share_feature
        proposal_detection_model.last_shared_layer_idx = max(shared_layer_idx);
        proposal_detection_model.last_shared_layer_detection = ...
            fast_rcnn_weights(proposal_detection_model.last_shared_layer_idx).layer_names;
        fprintf('please modify %s file for sharing conv layers with proposal model (delete layers until %s)\n', ...
            proposal_detection_model.detection_net_def, proposal_detection_model.last_shared_layer_detection);
    end
    
    save(fullfile(cachedir, 'model'), 'proposal_detection_model');
end

function [is_share_feature, proposal_weights, fast_rcnn_weights, shared_layer_idx] = check_proposal_fast_rcnn_model(proposal_model_net, proposal_model_bin, ...
        fast_rcnn_model_net, fast_rcnn_model_bin)

    init_net(fast_rcnn_model_net, fast_rcnn_model_bin, '//log//', 0);
    init_net(proposal_model_net, proposal_model_bin, '//log//', 1);
    fast_rcnn_weights = caffe('get_weights', 0);
    proposal_weights = caffe('get_weights', 1);
    
    is_share_feature = true;
    shared_layer_idx = [];
    for i = 1:min(length(fast_rcnn_weights), length(proposal_weights))
       if ~strcmp(fast_rcnn_weights(i).layer_names, proposal_weights(i).layer_names)
           break;
       end
       
       if ~isequal(fast_rcnn_weights(i).weights, proposal_weights(i).weights)
           is_share_feature = false;
       else
           shared_layer_idx(end+1) = i;
       end
    end
    
    caffe('release', 0);
    caffe('release', 1);
end

function [test_net_def_file, output_model_file] = find_last_output_model_file(stage1, stage2)
    if isfield(stage2, 'output_model_file') && exist(stage2.output_model_file, 'file')
        output_model_file = stage2.output_model_file;
        test_net_def_file = stage2.test_net_def_file;
        return;
    end
    if isfield(stage1, 'output_model_file') && exist(stage1.output_model_file, 'file')
        output_model_file = stage1.output_model_file;
        test_net_def_file = stage1.test_net_def_file;
        return;
    end
    error('find_last_output_model_file:: no trained models');
end