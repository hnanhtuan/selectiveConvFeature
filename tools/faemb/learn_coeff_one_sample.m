function [gama] = learn_coeff_one_sample(x, V, mu)
%Input:
%x: R^(D,1): a training samples
%V: R^(D,n): n bases, each base Vj in R^(D,1) 
%Goal:
% Compute coefficient gamma: R^(n,1) of a training sample x, given V

n = size(V,2); %number of bases
tmp = bsxfun(@minus, x, V);
A = sum(abs(tmp),1).^3;  %norm_1(xi - vj)^3

%% Compute Gamma
In = ones(n,1);

a = sum(A);
INV = inv(V'*V + a*mu*eye(n));
lambda = (In'  *  INV * V' * x - 1)/(In' * INV * In);
gama = INV * (V' * x - lambda * In);

