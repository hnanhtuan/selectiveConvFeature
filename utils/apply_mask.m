function [ masked_fea ] = apply_mask( filename, maskmethod )
% apply_mask
% + Load the feature map file.
% + Create the mask then apply
% + Return a set of local features

l = load(filename);
k = size(l.fea, 3);

switch maskmethod
    case 'max'
        mask = create_max_mask( l.fea );
    case {'sum50', 'sum'}
        mask = create_sum_mask( l.fea );
    case 'none'
        mask = true(size(l.fea, 1), size(l.fea, 2));
end

mask = repmat(mask, [1, 1, k]);
masked_fea =  l.fea(mask);
masked_fea = reshape(masked_fea, [length(masked_fea)/k, k])';
end

function [ mask ] = create_max_mask( fea )
% Return a binary mask

mask = false(size(fea, 1), size(fea, 2));
for j=1:size(fea, 3)
    [v1, p1] = max(fea(:, :, j));
    [~, p2] = max(v1);
    mask(p1(p2), p2) = 1;
end
mask = mask(:);
end


function [ mask ] = create_sum_mask( fea )
% Return a binary mask
mask = sum(fea, 3);
mask = (mask(:) >= prctile(mask(:), 50));
end

