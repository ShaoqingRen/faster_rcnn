
cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

fprintf('Downloading SPP_net_release1_VOC2007_selective_search_data...\n');
urlwrite('https://onedrive.live.com/download?resid=4006CBB8476FF777!9457&authkey=!AMB94UkybRV_pNs&ithint=file%2czip', ...
    'SPP_net_release1_VOC2007_selective_search_data.zip');

fprintf('Unzipping...\n');
unzip('SPP_net_release1_VOC2007_selective_search_data.zip', '.');

fprintf('Done.\n');
system('del SPP_net_release1_VOC2007_selective_search_data.zip');

cd(cur_dir);