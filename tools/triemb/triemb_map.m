function Xe = triemb_map (X, C, Pemb, Xm, slicesize)

if ~exist('slicesize')
  slicesize = 10000;
end

n = size(X, 2);
dout = size (Pemb, 1);

Xe = zeros(dout, n, 'single');

for i = 1:slicesize:n
  lasti = min (i+slicesize-1, n);
%   [Y, ~] = triemb_res(X(:,i:lasti), C, Xm);
  Y = triemb_res(X(:,i:lasti), C, Xm);
  Y = Pemb * Y;
  Xe (:, i:lasti) = Y;
end  

