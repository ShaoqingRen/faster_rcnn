function active_cvmex()
% active_cvmex()
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
    cur_dir = pwd;
    cd(fullfile(pwd, 'utils', 'opencv'));
    cv.imencode('.png', zeros(100, 100, 3));
    cd(cur_dir);
end