
cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

try
    fprintf('Downloading caffe_mex...\n');
    urlwrite('https://onedrive.live.com/download?resid=36FEC490FBC32F1A!111&authkey=!AFVWFGTbViiX5tg&ithint=file%2czip', ...
        'caffe_mex.zip');

    fprintf('Unzipping...\n');
    unzip('caffe_mex.zip', '..');

    fprintf('Done.\n');
    delete('caffe_mex.zip');
catch
    fprintf('Error in downloading, please try links in README.md https://github.com/ShaoqingRen/faster_rcnn'); 
end

cd(cur_dir);
