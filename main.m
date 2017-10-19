%% 
% Go to 'opt.m' file for change parameters.
opt;

%% Learning parameters
disp(['Test - ', dataset_name, ' - ', enc_method, ' - ', mask_method]);

param_file = [work_dir, dataset_name, '_param_', num2str(param.k), ...
                     '_', num2str(param.d), filename_surfix, '.mat'];
                 d
vecs_train_file = [dataset_dir, 'vecs_train_', num2str(param.k), '_', ...
                                    num2str(param.d),  filename_surfix, '.mat'];
if (exist(param_file, 'file') && ~overwrite_olddata)
    fprintf(2, ' * Load pretrained parameters!\n');
    load(param_file);
else
    fprintf(2, ' * Learning parameters ... \n');
    tic
    % Read all the feature map file in 'trainSetDir' and then applying mask
    gnd_train.imlist = dir([trainset_dir, '*.mat']);
    fea_train = cell(length(gnd_train.imlist), 1);
    parfor i=1:length(gnd_train.imlist)
        fea_train{i} = apply_mask([trainset_dir, gnd_train.imlist(i).name], mask_method);
    end

    % Apply PCA to reduce data-dimension
    vtrain = cell2mat(fea_train');
    param.desc_mean = mean(vtrain, 2);
    vtrain = bsxfun (@minus, vtrain, param.desc_mean);
    Xcov = vtrain * vtrain';
    Xcov = (Xcov + Xcov') / (2 * size (vtrain, 2));     % make it more robust
    [param.U, param.S, ~] = svd( Xcov );
    clear Xcov
    param.Ud = param.U(:,1:param.d);
    vtrain = param.Ud' * vtrain;                        % PCA
    vtrain = yael_vecs_normalize(vtrain);               % L2 normalize
    
    switch enc_method
        case {'temb', 'tembsink', 'tembmax'}
            param.C = yael_kmeans (vtrain, param.k, 'init', 1, 'redo', 2, 'niter', 100);
            [param.Xmean, param.eigvec, param.eigval] = triemb_learn (vtrain, param.C);
            param.eigval (end-32:end) = param.eigval (end-32);
            param.Pemb = diag(param.eigval.^-0.5) * param.eigvec';    % PCA-whitening
            
        case 'faemb'
            fprintf('Embedding method: FastFA-L1++\n');
            T          = 5;
            hes        = 1; 
            param.mu   = 10^-2;    % mu = 10^-2 is the recommended value
            [stat, B_init, f_total_init_best ] = ...
                    init_B_by_kmeans( vtrain, param.k, T, param.mu, emb_method);
            [gama_all] = learn_coeff_all_sample( vtrain, stat.B, param.mu, 'train');
            stat.gama_all_final = gama_all;

            [param.Xmean, param.eigvec, param.eigval] = ...
                    projection_learn_batch( vtrain, stat.B, stat.gama_all_final, param.k, hes);
            param.B = stat.B;
            clear vtrain stat gama_all

            % Compute whitening matrix to whiten phi(x)
            param.eigval(end-128:end) = param.eigval (end-128);
            param.Pemb = diag(single(param.eigval(1:end)).^-0.5) * single(param.eigvec(:,1:end))';
    end
    clear vtrain
    
    vecs_train = cell(length(gnd_train.imlist), 1);
    parfor i=1:length(gnd_train.imlist)
        vecs_train{i} = vecpostproc(embedding(fea_train{i}, param, enc_method));
    end
    clear fea_train
    
    % Learn RN
    vecs_train = cell2mat(vecs_train');
    param = learn_rn(vecs_train, param); 
    clear vecs_train
    
    if (save_param), save (param_file, 'param', '-v7.3'); end;
    fprintf ('Embedding parameters learned in %.3fs\n', toc);
end

%% Process database images
vecs_base_file = [dataset_dir, 'vecs_base_', num2str(param.k), '_', ...
            num2str(param.d),  filename_surfix, '.mat'];

if (exist(vecs_base_file, 'file') && ~overwrite_olddata)
    fprintf(2, ' * Load database image representation.\n');
    load(vecs_base_file);
else   
    fprintf(2, ' * Processing database images ... \n');
    tic
    vecs_base = cell(length(gnd_test.imlist), 1);
    parfor i=1:length(gnd_test.imlist)
        masked_fea   = apply_mask([baseset_dir, gnd_test.imlist{i}, '.mat'], mask_method);
        masked_fea   = vecpostproc(embedding(masked_fea, param, enc_method));
        masked_fea   = param.P' * bsxfun(@minus, masked_fea, param.Xm);
        vecs_base{i} = masked_fea(1 + truncate:end, :);
    end
    if (save_data), save(vecs_base_file, 'vecs_base', '-v7.3'); end;
    fprintf ('Embedding database in %.3fs (%.3fs/sample)\n', toc, toc/length(gnd_test.imlist));
end

%% Process query images
if (strcmp(dataset, 'ukb'))
    qvecs = vecs_base;
else
    qvecs_file = [dataset_dir, 'rqvecs_',  num2str(param.k), '_', ...
               num2str(param.d),  filename_surfix, '.mat'];
    
    if (exist(qvecs_file, 'file') && ~overwrite_olddata)
        fprintf(2, ' * Load query image representation.\n');
        load(qvecs_file);
    else   
        fprintf(2, ' * Processing query images ... \n');
        tic
        qimlist = {gnd_test.imlist{gnd_test.qidx}};
        qvecs = cell(length(qimlist), 1);
        parfor i=1:length(qimlist)
            masked_fea = apply_mask([queryset_dir, qimlist{i}, '.mat'], mask_method);
            masked_fea = vecpostproc(embedding(masked_fea, param, enc_method));
            masked_fea = param.P' * bsxfun(@minus, masked_fea, param.Xm);
            qvecs{i}   = masked_fea(1 + truncate:end, :);
        end
        if (save_data), save(qvecs_file, 'qvecs', '-v7.3'); end;
        fprintf ('Embedding query set in %.3fs (%.3fs/sample)\n', toc, toc/length(qimlist));
    end
end

%% Process flickr100k images
if (strcmp(dataset, 'oxford105k') || strcmp(dataset, 'paris106k'))
    
    vecs_flickr_file = [dataset_dir, 'vecs_flickr_', num2str(param.k), '_', ...
                num2str(param.d),  filename_surfix, '.mat'];
    if (exist(vecs_flickr_file, 'file') && ~overwrite_olddata)
        fprintf(2, ' * Load flickr image representation.\n');
        load(vecs_flickr_file);
    else   
        fprintf(2, ' * Processing flickr images ... \n');
        tic
        flickr_imlist = dir([flickrset_dir, '*.mat']);
        vecs_flickr   = cell(length(flickr_imlist), 1);
        parfor i=1:length(flickr_imlist)
            masked_fea     = apply_mask([flickrset_dir, flickr_imlist(i).name], mask_method);
            masked_fea     = vecpostproc(embedding(masked_fea, param, enc_method));
            masked_fea     = param.P' * bsxfun(@minus, masked_fea, param.Xm);
            vecs_flickr{i} = masked_fea(1 + truncate:end, :);
        end
        if (save_data), save(vecs_flickr_file, 'vecs_flickr', '-v7.3'); end;
        fprintf ('Embedding flickr in %.3fs (%.3fs/sample)\n', toc, toc/length(flickr_imlist));
    end
end

%% Evaluate
fprintf(2, ' * Evaluate Retrieval Performance\n');

% final database vectors and query vectors
vecs_base = cell2mat(vecs_base');
qvecs = cell2mat(qvecs');
if (strcmp(dataset, 'oxford105k') || strcmp(dataset, 'paris106k'))
    vecs_flickr = cell2mat(vecs_flickr');
    vecs_base = [vecs_base, vecs_flickr];
    clear vecs_flickr
end

for pw=[1 0.7 0.5 0.3 0.2 0]
    % Apply power-law normalization
    x = (sign(vecs_base) .* abs(vecs_base).^pw);
    q = (sign(qvecs) .* abs(qvecs).^pw);
    
    % l2 normalize to achieve the final vector
    x = yael_vecs_normalize (x, 2, 0);
    q = yael_vecs_normalize (q, 2, 0);

    if (~strcmp(dataset, 'ukb'))
        % retrieval with inner product
        [ranks,~] = yael_nn(x, -q, size(x, 2), 16);
        map = compute_map (ranks, gnd_test.gnd);

        % Query Expansion
        if (use_qe)
            qe = zeros(size(q), 'single');
            parfor qidx = 1:size(q, 2)
                qe(:, qidx) = mean([x(:, ranks(1:num_qe, qidx)), q(:, qidx)], 2);
            end
            [ranks_qe,~] = yael_nn(x, -qe, size(x, 2), 16);
            map_qe = compute_map (ranks_qe, gnd_test.gnd);
            fprintf ('%s  %s  %s  k=%d   d=%3d  D=%5d   pw=%.2f  mAP=%.3f mAP QE(%d)=%0.3f\n', ...
                   dataset_name, mask_method, enc_method, param.k, param.d, size(x,1), pw, map, num_qe, map_qe);
        else
            fprintf ('%s  %s  %s  k=%d   d=%3d  D=%5d   pw=%.2f  mAP=%.3f\n', ...
                   dataset_name, mask_method, enc_method, param.k, param.d, size(x,1), pw, map);
        end
    else
        [ranks,~] = yael_nn(x, -q, 4, 16);
        recall  = ukb_recall(ranks);
        fprintf ('%s  %s  %s  k=%d   d=%3d  D=%5d   pw=%.2f  recall@4=%.3f\n', ...
               dataset_name, mask_method, enc_method, param.k, param.d, size(x,1), pw, recall);
    end
end
fprintf(2, '================================================================\n');