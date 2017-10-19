% Learn the embedding parameters for triangulation embedding
%
% Usage:  [Xmean, eigvec, eigval] = triemb_learn (vtrain, C, dout)
%   vtrain    vector set for learning
%   C         centroids
%   dout      request output dimensionality
function [Xmean, eigvec, eigval] = triemb_learn (vtrain, C)

nlearn = size (vtrain, 2);     % number of input vectors
k = size (C, 2);               % number of support centroids
d = size (vtrain, 1);          % input vector dimensionality
D = k * d;                     % output dimensionality
dout = D;

slicesize = 10000;
nslices = nlearn / slicesize;
Xmean = zeros (D, 1, 'single');

% Compute mean embedded vector
Xsum = zeros (D, 1);
for i=1:slicesize:nlearn
  endi = min(i+slicesize-1, nlearn);
  X = triemb_res (vtrain (:,i:endi), C, Xmean);
  Xsum = Xsum + sum (X, 2);
end
Xmean = Xsum / nlearn;

% Compute whitening parameters
covD = zeros(d * k);
for i=1:slicesize:nlearn
  endi = min(i+slicesize-1, nlearn);
  X = triemb_res (vtrain (:,i:endi), C, Xmean);
  covD = covD + X * X';
end
fprintf ('\n');

% Eigen-decomposition
if 3 * dout < D
  eigopts.issym = true;
  eigopts.isreal = true;
  eigopts.tol = eps;
  eigopts.disp = 0;

  [eigvec, eigval] = eigs (double(covD), dout, 'LM', eigopts);
else
  [eigvec, eigval] = eig (covD);
  eigvec = eigvec (:, end:-1:end-dout+1);
  eigval = diag (eigval);
  eigval = eigval (end:-1:end-dout+1);
end
