function transformed_box = proposal_kmeans_box_transform(method, box)
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
    
    ctr_x = (box(:, 1) + box(:, 3)) * 1/2;
    ctr_y = (box(:, 2) + box(:, 4)) * 1/2;
    w = box(:, 3) - box(:, 1);
    h = box(:, 4) - box(:, 2);
    
    transformed_box = [ctr_x, ctr_y, ...
        c./(e+w), c./(e+h)];
end