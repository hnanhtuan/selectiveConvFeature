function [ param ] = learn_rn( L, param )

%---- Residual normalization
% Restrict PCA for optimization purpose
D = size (L, 1);

dpca = 1024;
dpca = min(dpca, D);

% Learn a pca - restrict learning to first eigenvalues
if dpca == 0, dpca = D; end

[~, eigvec, eigval, param.Xm] = yael_pca (L, dpca, true);

% Complement this with other vectors
P = [eigvec eye(D, D-size(eigvec, 2))];
eigval = [eigval ; eigval(end)*ones(D-size(eigvec, 2), 1)];
[param.P, ~] = qr (P);

end

