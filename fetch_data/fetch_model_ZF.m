
cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

try
    fprintf('Downloading model_ZF...\n');
    urlwrite('https://onedrive.live.com/download?resid=4006CBB8476FF777!17256&authkey=!AF7wGc1kbUTfI7o&ithint=file%2czip', ...
        'model_ZF.zip');

    fprintf('Unzipping...\n');
    unzip('model_ZF.zip', '..');

    fprintf('Done.\n');
    delete('model_ZF.zip');
catch
    fprintf('Error in downloading, please try links in README.md https://github.com/ShaoqingRen/faster_rcnn'); 
end

cd(cur_dir);
