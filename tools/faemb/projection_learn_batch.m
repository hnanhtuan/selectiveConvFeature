%% Function for learning projection params used for whitening stage

function [Xmean, eigvec, eigval] = projection_learn_batch (X, B, gama_all_final, k, hes)

if (~exist('hes', 'var'))
    hes = 1;
end

m = size (X, 2) ;    % number of input vectors
n = size(X, 1)   ;   % dimesion of input vectors
D = n * (n+1) * k / 2 ;          % input vector dimensionality
dout = D; % output dimensionality

slicesize = 10^4; % we can adjust this param correspond to the capability of memory's size for faster computing
% e.g. RAM = 64 GB --> slicesize = 10^5, etc.

Xsum = zeros (D, 1);
% fprintf(2,'Computing Xmean...\n\n');

for i=1:slicesize:m
    endi = min(i+slicesize-1, m);
%     fprintf ('\r%d-%d/%d', i, endi, m);
    
    %tmp = embeding( X(:,i:endi), B, gama_all_final(:,i:endi), hes );
    tmp = fa_embedding (single(X(:,i:endi)), single(B), single(gama_all_final(:,i:endi)));
    
    Xsum = Xsum + sum (tmp, 2);
end
Xmean = Xsum / m;
% fprintf(2,'\nFinish computing Xmean!\n\n');

% Compute whitening parameters
covD = zeros(D);

% fprintf(2,'\nComputing covariance matrix...\n\n');

for i=1:slicesize:m

  endi = min(i+slicesize-1, m);
%   fprintf ('\r%d-%d/%d', i, endi, m);

%   [ PHIX ] = embeding( X(:,i:endi), B, gama_all_final(:,i:endi), hes );
  [ PHIX ] = fa_embedding( single(X(:,i:endi)), single(B), single(gama_all_final(:,i:endi) ));
  
  PHIX = bsxfun(@minus, PHIX, Xmean);

  covD = covD + PHIX * PHIX';

end
covD = covD/(m-1);
% fprintf(2,'\nFinish computing covariance matrix!\n\n');

[eigvec, eigval] = eig (covD); %output by increasing orders of eigenvalue
eigvec = eigvec (:, end:-1:end-dout+1); %sort to decreasing orders
eigval = diag (eigval);
eigval = eigval (end:-1:end-dout+1);


% fprintf('\nLearning projection matrix finished!\n\n');
