function [im, im_scale] = prep_im_for_blob(im, im_means, target_size, max_size)
    im = single(im);
    try
        im = bsxfun(@minus, im, im_means);
    catch
        im_means = imresize(im_means, [size(im, 1), size(im, 2)], 'bilinear', 'antialiasing', false);    
        im = bsxfun(@minus, im, im_means);
    end
    im_scale = prep_im_for_blob_size(size(im), target_size, max_size);
       
    target_size = round([size(im, 1), size(im, 2)] * im_scale);
    im = imresize(im, target_size, 'bilinear', 'antialiasing', false);
end