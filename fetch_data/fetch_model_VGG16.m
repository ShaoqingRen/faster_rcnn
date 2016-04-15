
cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

try
    fprintf('Downloading model_VGG16...\n');
    urlwrite('https://onedrive.live.com/download?resid=36FEC490FBC32F1A!114&authkey=!AE8uV9B07dREbhM&ithint=file%2czip', ...
        'model_VGG16.zip');

    fprintf('Unzipping...\n');
    unzip('model_VGG16.zip', '..');

    fprintf('Done.\n');
    delete('model_VGG16.zip');
catch
    fprintf('Error in downloading, please try links in README.md https://github.com/ShaoqingRen/faster_rcnn');
end

cd(cur_dir);
