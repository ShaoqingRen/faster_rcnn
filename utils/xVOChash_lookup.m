function ind = xVOChash_lookup(hash,s)
% From the PASCAL VOC 2011 devkit

hsize=numel(hash.key);
h=mod(str2double(s([4 6:end])),hsize)+1;
ind=hash.val{h}(strmatch(s,hash.key{h},'exact'));
