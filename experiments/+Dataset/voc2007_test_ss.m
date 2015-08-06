function dataset = voc2007_test_ss(dataset, usage, use_flip)
% Pascal voc 2007 test set with selective search
% set opts.imdb_train opts.roidb_train 
% or set opts.imdb_test opts.roidb_train

% change to point to your devkit install
devkit                      = voc2007_devkit();

switch usage
    case {'train'}
        dataset.imdb_train    = {  imdb_from_voc(devkit, 'test', '2007', use_flip) };
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x, 'with_selective_search', true), dataset.imdb_train, 'UniformOutput', false);
    case {'test'}
        dataset.imdb_test     = imdb_from_voc(devkit, 'test', '2007', use_flip) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, 'with_selective_search', true);
    otherwise
        error('usage = ''train'' or ''test''');
end

end