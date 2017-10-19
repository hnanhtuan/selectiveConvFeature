function [ recall ] = ukb_recall( ranks )

recalls = zeros(size(ranks, 2), 1);
for i=1:size(ranks, 2)
    R = mod(i, 4);
    Q = (i - R)/4;
    if (R == 0), Q = Q - 1; end;
    start_idx = Q*4 + 1;
    end_idx = Q*4 + 4;
    
    recalls(i) = sum((ranks(:, i) >= start_idx) & (ranks(:, i) <= end_idx));
end
recall = mean(recalls);
end

