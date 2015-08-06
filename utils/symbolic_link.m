function symbolic_link(link, target)
% symbolic_link(link, target)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    if ispc()
        system(sprintf('mklink /J %s %s', link, target)); 
    else 
        system(sprintf('ln -s %s %s', link, target)); 
    end

end
