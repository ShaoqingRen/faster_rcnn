function model = ZF_for_Fast_RCNN_VOC2007(model)

model.solver_def_file        = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'ZF', 'solver_30k40k.prototxt');
model.test_net_def_file      = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'ZF', 'test.prototxt');

model.net_file               = fullfile(pwd, 'models', 'pre_trained_models', 'ZF', 'ZF.caffemodel');
model.mean_image             = fullfile(pwd, 'models', 'pre_trained_models', 'ZF', 'mean_image');

end