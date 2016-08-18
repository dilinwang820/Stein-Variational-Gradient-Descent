function [Akxy, info] = KSD_KL_gradxy(x, dlog_p, h)
%%%%%%%%%%%%%%%%%%%%%%
% Input:
%    -- x: particles, n*d matrix, where n is the number of particles and d is the dimension of x 
%    -- dlog_p: a function handle, which returns the first order derivative of log p(x), n*d matrix
%    -- h: bandwidth. If h == -1, h is selected by the median trick

% Output:
%    --Akxy: n*d matrix, \Phi(x) is our algorithm, which is a smooth
%    function that characterizes the perturbation direction
%    --info: kernel bandwidth
%%%%%%%%%%%%%%%%%%%%%%

if nargin < 3; h = -1; end % median trick as default

[n, d] = size(x);

%%%%%%%%%%%%%% Main part %%%%%%%%%%
Sqy = dlog_p(x);

% Using rbf kernel as default
XY = x*x';
x2= sum(x.^2, 2);
X2e = repmat(x2, 1, n);

H = (X2e + X2e' - 2*XY); % calculate pairwise distance

% median trick for bandwidth
if h == -1
    h = sqrt(0.5*median(H(:)) / log(n+1));   %rbf_dot has factor two in kernel
end

Kxy = exp(-H/(2*h^2));   % calculate rbf kernel


dxKxy= -Kxy*x;
sumKxy = sum(Kxy,2);
for i = 1:d
    dxKxy(:,i)=dxKxy(:,i) + x(:,i).*sumKxy;
end
dxKxy = dxKxy/h^2;
Akxy = (Kxy*Sqy + dxKxy)/n;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

info.bandwidth = h;

return;
end

