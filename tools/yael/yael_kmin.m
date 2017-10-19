% This function returns the k smallest values of a vector 
%
% Usage: [val, idx] = yael_kmin (v,k)
%
% Parameters:
%   v     the vector to be partially ranked. If v is a matrix, the function
%         returns the k largest values of each column (like min function)
%   k     the number of neighbors to be returned. Must be smaller than vector length
%
% Output:
%   val   a k-dimensional vector containing the (ordered) set of smallest values
%         In case v was a matrix, val is a k*n matrix with one column per vector
%   idx   the indexes, in the original vector, where the smallest values have been found
%         This output parameter is not mandatory
%
% Remarks: if k=1, this function is equivalent to searching the min. 
%          if k is equal to the vector length, it is equivalent to the sort function
function [val, idx] = yael_kmin (v, k)

fprintf ('# Warning: This is NOT the fast implementation. \n#You should use the Mex version instead\n');

[val, idx] = sort (v);

val = val (1:k, :);
idx = idx (1:k, :);
