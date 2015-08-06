function roidb_new = do_proposal_test(conf, model_stage, imdb, roidb)
    aboxes                      = proposal_test(conf, imdb, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name);      
                                    
    aboxes                      = boxes_filter(aboxes, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN, conf.use_gpu);    
    
    roidb_regions               = make_roidb_regions(aboxes, imdb.image_ids);  
    
    roidb_new                   = roidb_from_proposal(imdb, roidb, roidb_regions, ...
                                        'keep_raw_proposal', false);    
end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    % to speed up nms
    if per_nms_topN > 0
        aboxes = cellfun(@(x) x(1:min(length(x), per_nms_topN), :), aboxes, 'UniformOutput', false);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        if use_gpu
            for i = 1:length(aboxes)
                aboxes{i} = aboxes{i}(nms(aboxes{i}, nms_overlap_thres, use_gpu), :);
            end 
        else
            parfor i = 1:length(aboxes)
                aboxes{i} = aboxes{i}(nms(aboxes{i}, nms_overlap_thres), :);
            end       
        end
    end
    aver_boxes_num = mean(cellfun(@(x) size(x, 1), aboxes, 'UniformOutput', true));
    fprintf('aver_boxes_num = %d, select top %d\n', round(aver_boxes_num), after_nms_topN);
    if after_nms_topN > 0
        aboxes = cellfun(@(x) x(1:min(length(x), after_nms_topN), :), aboxes, 'UniformOutput', false);
    end
end

function regions = make_roidb_regions(aboxes, images)
    regions.boxes = aboxes;
    regions.images = images;
end
