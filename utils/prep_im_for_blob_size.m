function im_scale = prep_im_for_blob_size(im_size, target_size, max_size)

    im_size_min = min(im_size(1:2));
    im_size_max = max(im_size(1:2));
    im_scale = double(target_size) / im_size_min;
    
    % Prevent the biggest axis from being more than MAX_SIZE
    if round(im_scale * im_size_max) > max_size
        im_scale = double(max_size) / double(im_size_max);
    end
end