function x = vecpostproc(x, a) 
	if ~exist('a'), a = 1; end
	x = replacenan (yael_vecs_normalize (powerlaw (x, a)));

% replace all nan values in a matrix (with zero)
function y = replacenan (x, v)

if ~exist ('v')
  v = 0;
end

y = x;
y(isnan(x)) = v;	

% apply powerlaw
function x = powerlaw (x, a)

if a == 1, return; end
x = sign (x) .* abs(x)  .^ a;