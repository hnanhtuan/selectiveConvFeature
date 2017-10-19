% Usage: Y = triemb_res (X, C, Xm)
% 
% Perform the triangulation embedding 
%    X   input vectors
%    C   centroids
%    Xm  mean to be removed 
function Y = triemb_res (X, C,  Xm)

n = size(X, 2);
d = size(X, 1);
kc = size(C, 2);
D = d * kc;

Y = bsxfun (@minus, repmat(X, kc, 1), C(:));

for j = 1:kc
  idxj = 1+(j-1)*d : j*d;
  Y(idxj, :) = yael_vecs_normalize (Y(idxj,:));
end

Y = bsxfun (@minus, Y, Xm);
