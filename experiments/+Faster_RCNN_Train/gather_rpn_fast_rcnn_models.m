function gather_rpn_fast_rcnn_models(conf_proposal, conf_fast_rcnn, model, dataset)
    cachedir = fullfile(pwd, 'output', 'faster_rcnn_final', model.final_model.cache_name);
    mkdir_if_missing(cachedir);
    
    % find latest model for rpn and fast rcnn
    [rpn_test_net_def_file, rpn_output_model_file] = find_last_output_model_file(model.stage1_rpn, model.stage2_rpn);
    [fast_rcnn_test_net_def_file, fast_rcnn_output_model_file] = find_last_output_model_file(model.stage1_fast_rcnn, model.stage2_fast_rcnn);
    
    % check whether feature shared and find the indexs of shared layers
    [is_share_feature, last_shared_output_blob_name, shared_layer_names, shared_layer_idx] = ...
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
            shared_layer_names{proposal_detection_model.last_shared_layer_idx};
        proposal_detection_model.last_shared_output_blob_name = ...
            last_shared_output_blob_name;
        fprintf('please modify %s file for sharing conv layers with proposal model (delete layers until %s)\n', ...
            proposal_detection_model.detection_net_def, proposal_detection_model.last_shared_layer_detection);
    end
    
    save(fullfile(cachedir, 'model'), 'proposal_detection_model');
end

function [is_share_feature, last_shared_output_blob_name, shared_layer_names, shared_layer_idx] = check_proposal_fast_rcnn_model(proposal_model_net, proposal_model_bin, ...
        fast_rcnn_model_net, fast_rcnn_model_bin)

    rpn_net = caffe.Net(proposal_model_net, 'test');
    rpn_net.copy_from(proposal_model_bin);
    
    fast_rcnn_net = caffe.Net(fast_rcnn_model_net, 'test');
    fast_rcnn_net.copy_from(fast_rcnn_model_bin);
    
    share_layer = true;
    shared_layer_idx = [];
    shared_layer_names = {};
    shared_rpn_blobs = {};
    for i = 1:min(length(rpn_net.layer_names), length(fast_rcnn_net.layer_names))
       if ~strcmp(rpn_net.layer_names{i}, fast_rcnn_net.layer_names{i})
           break;
       end
       
       rpn_layer_name = rpn_net.layer_names{i};
       fast_rcnn_layer_name = fast_rcnn_net.layer_names{i};
       rpn_layer = rpn_net.layers(rpn_layer_name);
       fast_rcnn_layer = fast_rcnn_net.layers(fast_rcnn_layer_name);
       
       for j = 1:min(length(rpn_layer.params), length(fast_rcnn_layer.params))
           if ~isequal(rpn_net.params(rpn_layer_name, j).get_data(), fast_rcnn_net.params(fast_rcnn_layer_name, j).get_data())
               share_layer = false;
           end 
       end
       
       if ~share_layer 
           break;
       else
           shared_layer_idx(end+1) = i;
           shared_layer_names{end+1} = rpn_layer_name; 
           last_shared_output_blob_name = rpn_net.blob_names{rpn_net.top_id_vecs{i}};
       end
    end
    
    is_share_feature = false;
    if ~isempty(shared_layer_idx)
        is_share_feature = true;
    end
    
    caffe.reset_all(); 
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