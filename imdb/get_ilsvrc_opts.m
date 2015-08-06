function ilsvrc_opts = get_ilsvrc_opts(path, onlyVOC20classes)

tmp = pwd;
cd(path);
try
  addpath(fullfile(pwd, 'evaluation'));
  if onlyVOC20classes
      ilsvrc_init_voc20classes;
  else
      ilsvrc_init;
  end
catch
  rmpath(fullfile(pwd, 'evaluation'));
  cd(tmp);
  error(sprintf('ilsvrc directory not found under %s', path));
end
cd(tmp);
