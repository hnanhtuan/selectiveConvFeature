% Compute the weights to tend towards a democratic kernel
% Usage: alpha = qdemocratic (x, method, tau)
%  method   'sum'
%           'sinkhorn'    regular Sinkhorn
%           'zca'         ZCA
%
%  tau       sparsificatino of the Gram matrix
%
%  param1, param2, ...    Method-dependent
%
function [y, alpha] = qdemocratic (x, method, param1, param2, param3)

% d = size (x, 1);
% n = size (x, 2);
% 
% alpha = [];
% 
% if n == 0
%   alpha = [];
%   y = zeros (d, 1);
%   return;
% end

%--------------------------------------
% if strcmp (method, 'sum')
%   alpha = ones (1, n);
%   y = sum (x, 2);
%   return;
% end


% if strcmp (method, 'sinkhorn')
  % Gram matrix
  G = x' * x ;

  % Filter out kernel elements below 0
  G = (G + abs(G)) / 2; % Faster than finding negative values and putting 0

%   if exist ('param1')  reg = param1;
%   else                 reg = 0.5;  end
  reg = 0.5;

%   if exist ('param2')  nbiter = param2;
%   else                 nbiter = 10;  end
  nbiter = 10;
  [~, alpha, ~] = sinkhornm(G, reg, nbiter);

  y = sum(bsxfun(@times, x, alpha), 2);
%   return
% end
