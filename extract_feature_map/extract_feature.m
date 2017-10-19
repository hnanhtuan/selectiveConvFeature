function [ X ] = extract_feature( I, net )

if size(I,3) == 1
  I = repmat(I, [1 1 3]);
end
I = single(I) - mean(net.meta.normalization.averageImage(:));

if ~isa(net.layers{1}.weights{1}, 'gpuArray')
    rnet = vl_simplenn(net, I);  
    X = rnet(end).x;
else
    rnet = vl_simplenn(net, gpuArray(I));  
    X = gather(rnet(end).x);
end
end

