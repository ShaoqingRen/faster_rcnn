function model = VGG16_for_Fast_RCNN_VOC2007(model)
% VGG 16layers (only finetuned from conv3_1)

model.solver_def_file        = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'vgg_16layers_conv3_1', 'solver_30k40k.prototxt');
model.test_net_def_file      = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'vgg_16layers_conv3_1', 'test.prototxt');

model.net_file               = fullfile(pwd, 'models', 'pre_trained_models', 'vgg_16layers', 'vgg16.caffemodel');
model.mean_image             = fullfile(pwd, 'models', 'pre_trained_models', 'vgg_16layers', 'mean_image');

end