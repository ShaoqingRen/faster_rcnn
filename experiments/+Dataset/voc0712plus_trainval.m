function dataset = voc0712plus_trainval(dataset, usage, use_flip)
% Pascal voc 0712 trainval set 
% set opts.imdb_train opts.roidb_train 
% or set opts.imdb_test opts.roidb_train

% change to point to your devkit install
devkit2007                      = voc2007_devkit();
devkit2012                      = voc2012_devkit();

switch usage
    case {'train'}
        dataset.imdb_train    = {  imdb_from_voc(devkit2012, 'trainval', '2012', use_flip), ... 
                                    imdb_from_voc(devkit2007, 'trainval', '2007', use_flip), ... 
                                    imdb_from_voc(devkit2007, 'test', '2007', use_flip)};
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x), dataset.imdb_train, 'UniformOutput', false);
    case {'test'}
        error('only supports one source test currently');  
    otherwise
        error('usage = ''train'' or ''test''');
end

end