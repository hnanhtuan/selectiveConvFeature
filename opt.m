addpath(genpath('tools/'));
addpath(genpath('utils/'));
addpath(genpath('data/'));

poolobj = gcp;
addAttachedFiles(poolobj, {'triemb_map.m', 'triemb_res.mexa64', ...
                            'qdemocratic.m', 'sinkhornm.m', ...
                            'embedding.m', 'vecpostproc.m'})

%% Dataset 
data_dir = 'data/';
work_dir = [data_dir, 'workdir/'];

dataset = 'paris6k'; 
switch dataset
    case {'oxford5k', 'oxford105k'}
        dataset_train				= 'paris6k';        % dataset to learn the PCA-whitening on
        dataset_test 				= 'oxford5k';       % dataset to evaluate on 
    case {'paris6k', 'paris106k'}
        dataset_train				= 'oxford5k';       % dataset to learn the PCA-whitening on
        dataset_test 				= 'paris6k';        % dataset to evaluate on 
    case 'holidays'
        dataset_train				= 'flickr5k';       % dataset to learn the PCA-whitening on
        dataset_test 				= 'holidays';       % dataset to evaluate on 
    case 'ukb'
        dataset_train               = 'flickr5k';       % dataset to learn the PCA-whitening on
        dataset_test                = 'ukb';            % dataset to evaluate on 
end
gnd_test = load(['gnd_', dataset_test, '.mat']);

lid         = 31;   % VGG layer Id 31 - 29 
max_img_dim = 1024;

% The 'dataset_name' should be the same folder where the extracted conv.
% features are stored.
dataset_name  = [dataset_test, '_', num2str(lid),'_', num2str(max_img_dim)];

dataset_dir   = [data_dir, dataset_name, '/'];
trainset_dir  = [dataset_dir, dataset_train, '/'];
baseset_dir   = [dataset_dir, dataset_test, '/'];
queryset_dir  = [dataset_dir, dataset_test, 'q/'];
flickrset_dir = [dataset_dir, 'flickr100k/'];

filename_surfix = [ '_', enc_method, '_', mask_method ];
disp(filename_surfix);
%% Parameters
enc_method      = 'temb';
    % 'temb':      Triangular embedding + Sum pooling
    % 'tembsink':  Triangular embedding + Democratic pooling
    % 'faemb':     Fast-Function Apprximate Embdding + Democratic pooling
mask_method     = 'max';    % 'max', 'sum50', 'none'

switch enc_method
    case {'temb', 'tembsink'}
        truncate        = 128;                  % Truncate the first 128 dimensions
    case 'faemb'
        truncate        = param.d*(param.d+1);  % Truncate the first d(d+1) dimensions
    otherwise
        error('Invalid Encoding method (Please choose one of following: "temb", "tembsink", "faemb")')
end

param.d         = 32;       % Remaining dimension of PCA pre-processing
param.k         = 20;        % Number of codebook size       

% Recommending values for retained PCA components and codebook size
% 'temb', 'tembsink'
%| Dim. D | 512-D | 1024-D | 2048-D | 4096-D | 8064-D |
%|--------|-------|--------|--------|--------|--------|
%| d      | 32    | 64     | 64     | 64     | 128    |
%| k      | 20    | 18     | 34     | 66     | 64     |
%
% 'faemb' https://www.dropbox.com/s/1dzxls5pplf8iuo/FAemb-TPAMI.pdf?dl=0
%| Dim. D | 4224-D |
%|--------|--------|
%| d      | 32     |
%| k      | 10     |

use_qe          = false;        % Set true to apply Query Expansion (QE)
num_qe          = 5;    

save_param = true;              % Save the learned parameters for later usage
save_data  = false;             % Save the processed global image representation for later usage
overwrite_olddata = true;       % Re-learn the parameters and re-process image (if did).


