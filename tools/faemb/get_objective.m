function [ f,   f_locality, f_reconstruction ] = get_objective( X, B, gama_all, mu )
%% Compute objective function value
% compute objective function 
% objective function: (1/2) * || B*gama_all - X ||^2 + 
%                     (mu/2) * sum_{i=1}^{m}  ||gama_i||^2 sum_{j=1}^{n} norm_1(x_i - v_j)^3  


m = size(X,2); %number of training samples
n = size(B,2); %number of bases

f_locality = 0;
res_X = B * gama_all - X;
f_reconstruction = 0.5 * sum(sum(res_X.^2, 1));

%objective function: (1/2) * || B*gama_all - X ||^2 + (mu/2) * sum_{i=1}^{m}  ||gama_i||^2 sum_{j=1}^{n} abs(x_i - v_j)^3
gama_norm_2 = sum(gama_all.^2,1);
for j = 1:n    
    bj  = B(:,j);
    resj = bsxfun(@minus, bj, X);
    resj_norm_3 = sum(abs(resj),1).^3;  %norm_1(v_j - x_i) ^ 3
    tmp2 = bsxfun(@times, gama_norm_2, resj_norm_3);
    f_locality = f_locality + sum(tmp2);    
end

f =  f_reconstruction + 0.5*mu*f_locality;
%divide by number of training samples
divide_by_number_of_samples = false;
if divide_by_number_of_samples
    f = f/m;
    f_locality = f_locality/m;
    f_reconstruction = f_reconstruction/m;
end