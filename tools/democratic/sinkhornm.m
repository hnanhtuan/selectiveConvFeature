% This is a modified version of the SINKHORN algorithm (see sinhorn.m)
% 
% It converts a square positive matrix to a matrix that sums to a constant C
% (rows and columns sum to one) based on the Knight variant.
% 
% Usage: [kn, d1, nbiter] = sinkhornm(kn, reg, nbiter)
%
% The algorithm steps after nbiter iterations

function [kn, lambda, nbiter] = sinkhornm(kn, reg, nbiter, verbose)

if ~exist ('reg', 'var'),    reg = 0.5; end
if ~exist ('nbiter', 'var'), nbiter = 10; end
if ~exist ('verbose', 'var'), verbose = false; end

n = size (kn, 1);
lambda = ones (1, size(kn, 1), 'single');

for k=1:nbiter
    delta = 1./(sum(kn).^reg);
    kn = bsxfun (@times, kn, delta);
    kn = bsxfun (@times, kn, delta');
    lambda = delta .* lambda;
end
