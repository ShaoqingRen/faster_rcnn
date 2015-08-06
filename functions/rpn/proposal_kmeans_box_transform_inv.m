function transformed_box = proposal_kmeans_box_transform_inv(method, box)
% transformed_box = proposal_kmeans_box_transform_inv(method, box)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   

    trans_fun = str2func(method);
    transformed_box = trans_fun(box);
end

function transformed_box = LTRB(box)
% both box and transformed_box are unit normalized 
    transformed_box = box;
end

function transformed_box = multibox(box)
% both box and transformed_box are unit normalized 
    c = 0.2;
    e = 0.01;
    
    ctr_x = box(:,1);
    ctr_y = box(:,2);
    w = c ./ (box(:, 3) + eps) - e;
    h = c ./ (box(:, 4) + eps) - e;
    
    transformed_box = [ctr_x - 0.5*(w), ctr_y - 0.5*(h), ...
                ctr_x + 0.5*(w), ctr_y + 0.5*(h)];
end