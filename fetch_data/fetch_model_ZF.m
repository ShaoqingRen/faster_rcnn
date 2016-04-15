
cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

try
    fprintf('Downloading model_ZF...\n');
    urlwrite('https://onedrive.live.com/download?resid=36FEC490FBC32F1A!113&authkey=!AIzdm0sD_SmhUQ4&ithint=file%2czip', ...
        'model_ZF.zip');

    fprintf('Unzipping...\n');
    unzip('model_ZF.zip', '..');

    fprintf('Done.\n');
    delete('model_ZF.zip');
catch
    fprintf('Error in downloading, please try links in README.md https://github.com/ShaoqingRen/faster_rcnn'); 
end

cd(cur_dir);
