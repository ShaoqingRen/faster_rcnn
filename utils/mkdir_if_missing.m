function made = mkdir_if_missing(path)
made = false;
if exist(path, 'dir') == 0
  mkdir(path);
  made = true;
end
