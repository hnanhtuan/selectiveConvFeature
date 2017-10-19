% This function reads a vector from a file in the libit format

function [v,d] = ivec_read (fid)

% first read the vector size
d = fread (fid, 1, 'int');

% read the elements
v = fread (fid, d, 'int');

