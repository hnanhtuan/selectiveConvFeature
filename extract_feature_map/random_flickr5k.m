base_dir = '/home/anhxtuan/Documents/Datasets/retrieval/';

in_folder  = [base_dir, 'flickr100k_31/'];
out_folder = [base_dir, 'flickr5k_31/'];
if (~exist(out_folder, 'dir')), mkdir(out_folder); end;

filelist = dir([in_folder, 'flickr100k/*.mat']);
idx = randperm(length(filelist));
filelist = filelist(idx', :);

filelist = filelist(1:5000);

for i=1:length(filelist)
    copyfile([in_folder, 'flickr100k/', filelist(i).name], [out_folder, 'flickr5k/', filelist(i).name]);
end