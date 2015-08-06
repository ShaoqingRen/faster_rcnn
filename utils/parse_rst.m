function results = parse_rst(results, rst)
% results = parse_rst(results, rst)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    if isempty(results)
        for i = 1:length(rst)
            results.(rst(i).blob_name).data = [];
        end
    end
        
    for i = 1:length(rst)
        results.(rst(i).blob_name).data = [results.(rst(i).blob_name).data; rst(i).data(:)];
    end
end