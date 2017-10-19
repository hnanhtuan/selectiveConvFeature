run /path/to/matconvnet-1.0-beta25/matlab/vl_setupnn.m

%% Parameters
modelfn     = 'imagenet-vgg-verydeep-16.mat';
lid         = 31;       % The index of conv. layer to extract features.
max_img_dim = 1024;     % Resize to have max(W, H)=max_img_dim
baseDir     = '/path/to/image/folder/';   % Image folder

net = load(modelfn);
net.layers = {net.layers{1:lid}}; % remove fully connected layers
net = vl_simplenn_tidy(net);
net = vl_simplenn_move(net, 'gpu') ;

folder_suffix = [num2str(lid),'_', num2str(max_img_dim)];

%% Select the dataset 
oxford_paris = true;
holidays     = false;
flickr100k   = false;
ukb          = false;

%%
if (oxford_paris)
im_folder_oxford  = [baseDir, 'oxford5k/'];
im_folder_paris   = [baseDir, 'paris6k/'];

out_folder_oxford = [baseDir, 'oxford5k_', folder_suffix, '/'];
out_folder_paris  = [baseDir, 'paris6k_', folder_suffix, '/'];
base_set_paris    = [out_folder_paris, 'paris6k/'];
query_set_paris   = [out_folder_paris, 'paris6kq/'];
base_set_oxford   = [out_folder_oxford, 'oxford5k/'];
query_set_oxford  = [out_folder_oxford, 'oxford5kq/'];

if (~exist(out_folder_oxford, 'dir')), mkdir(out_folder_oxford); end;
if (~exist(out_folder_paris, 'dir')),  mkdir(out_folder_paris);  end;
if (~exist(base_set_paris, 'dir')),    mkdir(base_set_paris);    end;
if (~exist(query_set_paris, 'dir')),   mkdir(query_set_paris);   end;
if (~exist(base_set_oxford, 'dir')),   mkdir(base_set_oxford);   end;
if (~exist(query_set_oxford, 'dir')),  mkdir(query_set_oxford);  end;

gnd_oxford = load('gnd_oxford5k.mat');
gnd_paris  = load('gnd_paris6k.mat');    

fprintf('Extracting features\n');

for i=1:length(gnd_paris.imlist)
    I = imread([im_folder_paris, gnd_paris.imlist{i}, '.jpg']);
    ratio = max_img_dim/max(size(I, 1), size(I, 2));
    I = imresize(I, ratio);
    
    fea = extract_feature( I, net );
    save([base_set_paris, gnd_paris.imlist{i}, '.mat'], 'fea');
    disp([num2str(i), ' --- ', gnd_paris.imlist{i}]);	
end

for i=1:length(gnd_oxford.imlist)
    I = imread([im_folder_oxford, gnd_oxford.imlist{i}, '.jpg']);
    ratio = max_img_dim/max(size(I, 1), size(I, 2));
    I = imresize(I, ratio);
    
    fea = extract_feature( I, net );
    save([base_set_oxford, gnd_oxford.imlist{i}, '.mat'], 'fea');
    disp([num2str(i), ' --- ', gnd_oxford.imlist{i}]);	
end

qimlist = {gnd_oxford.imlist{gnd_oxford.qidx}};
for i=1:length(qimlist)
    I = imread([im_folder_oxford, qimlist{i}, '.jpg']);
    ratio = max_img_dim/max(size(I, 1), size(I, 2));
    I = crop_qim([im_folder_oxford, qimlist{i}, '.jpg'], gnd_oxford.gnd(i).bbx);
    I = imresize(I, ratio);
    
    fea = extract_feature( I, net );
    save([query_set_oxford, qimlist{i}, '.mat'], 'fea');
    disp([num2str(i), ' --- ', qimlist{i}]);	
end

qimlist = {gnd_paris.imlist{gnd_paris.qidx}};
for i=1:length(qimlist)
    I = imread([im_folder_paris, qimlist{i}, '.jpg']);
    ratio = max_img_dim/max(size(I, 1), size(I, 2));
    I = crop_qim([im_folder_paris, qimlist{i}, '.jpg'], gnd_paris.gnd(i).bbx);
    I = imresize(I, ratio);
    
    fea = extract_feature( I, net );
    save([query_set_paris, qimlist{i}, '.mat'], 'fea');
    disp([num2str(i), ' --- ', qimlist{i}]);	
end
end

%%
if (ukb)
    im_folder  = [baseDir, 'UKB/'];
    out_folder = [baseDir, 'ukb_', folder_suffix, '/'];
    base_set   = [out_folder, 'ukb/'];
    if (~exist(out_folder, 'dir')), mkdir(out_folder); end;
    if (~exist(base_set, 'dir')),   mkdir(base_set); end;
    
    imlist = dir([im_folder, '*.jpg']);
    for i=1:length(imlist)
        orig = imread([im_folder, imlist(i).name]);
        ratio = max_img_dim/max(size(orig, 1), size(orig, 2));
        I = imresize(orig, ratio);
        fea = extract_feature( I, net );
        save([base_set, strrep(imlist(i).name, '.jpg', '.mat')], 'fea');
        disp([num2str(i), ' --- ', imlist(i).name]);	
    end
end

%% Holidays
if (holidays)
im_folder  = [baseDir, 'holidays/'];
out_folder = [baseDir, 'holidays_', folder_suffix, '/'];
base_set   = [out_folder, 'holidays/'];
if (~exist(out_folder, 'dir')), mkdir(out_folder); end;
if (~exist(base_set, 'dir')),   mkdir(base_set); end;

imlist     = dir([im_folder, '*.jpg']);

for i=1:length(imlist)
    orig = imread([im_folder, imlist(i).name]);
    ratio = max_img_dim/max(size(orig, 1), size(orig, 2));
    I = imresize(orig, ratio);
    fea = extract_feature( I, net );
    save([base_set, strrep(imlist(i).name, '.jpg', '.mat')], 'fea');
    disp([num2str(i), ' --- ', imlist(i).name]);	
end
end
%% 
if(flickr100k)
    im_folder  = [baseDir, 'flickr100k_jpg/oxc1_100k/'];
    out_folder = [baseDir, 'flickr100k_', folder_suffix, '/'];
    base_set   = [out_folder, 'flickr100k/'];
    if (~exist(out_folder, 'dir')), mkdir(out_folder); end;
    if (~exist(base_set, 'dir')),   mkdir(base_set);   end;
    
    subDirList = dir([im_folder]);
    subDirList(1:2) = [];
    for i=1:length(subDirList)
        imlist = dir([im_folder, subDirList(i).name, '/*.jpg']);
        for j=1:length(imlist)
            try
                I = imread([im_folder, subDirList(i).name, '/', imlist(j).name]);
                if (size(I, 3) ~= 3), continue; end;
                ratio = max_img_dim/max(size(I, 1), size(I, 2));
                I = imresize(I, ratio);
                fea = extract_feature( I, net );
                save([out_folder, strrep(imlist(j).name, '.jpg', '.mat')], 'fea');

                disp([num2str(j), ' --- ', num2str(i), ' --- ' , imlist(j).name]);
            catch
                delete([im_folder, subDirList(i).name, '/', imlist(j).name]);
            end
        end
    end
end