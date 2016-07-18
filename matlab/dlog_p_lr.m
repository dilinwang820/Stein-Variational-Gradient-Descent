function dlog_p = dlog_p_lr(theta, X, Y, batchsize, a0, b0)
%%%%%%%
% Output: First order derivative of Bayesian logistic regression. 
        
% The inference is applied on posterior p(theta|X, Y) with theta = [w, log(alpha)], 
% where p(theta|X, Y) is the bayesian logistic regression 
% We use the same settings as http://icml.cc/2012/papers/360.pdf

% When the number of observations is very huge, computing the derivative of
% log p(x) could be the major computation bottleneck. We can conveniently
% address this problem by approximating with subsampled mini-batches

% Input:
%   -- theta: a set of particles, M*d matrix (M is the number of particles)
%   -- X, Y: observations, where X is the feature matrix and Y contains
%   target label
%   -- batchsize, sub-sampling size of each batch;batchsize = -1, calculating the derivative exactly
%   -- a0, b0: hyper-parameters
%%%%%%%

[N, ~] = size(X);  % N is the number of total observations

if nargin < 4; batchsize = min(N, 100); end % default batch size 100
if nargin < 4; a0 = 1; end
if nargin < 5; b0 = 1; end

if batchsize  > 0
    ridx = randperm(N, batchsize);
    X = X(ridx,:); Y = Y(ridx,:);  % stochastic version
end

w = theta(:, 1:end-1);  %logistic weights
alpha = exp(theta(:,end)); % the last column is logalpha
D = size(w, 2);

wt = (alpha/2).*(sum(w.*w, 2));
y_hat = 1./(1+exp(-X*w'));

dw_data = ((repmat(Y,1,size(theta,1))+1)/2 - y_hat)' * X; % Y \in {-1,1}
dw_prior = - repmat(alpha,1,D) .* w;
dw = dw_data * N /size(X,1) + dw_prior; %re-scale

dalpha = D/2 - wt + (a0-1) - b0.*alpha + 1;  %the last term is the jacobian term

dlog_p = [dw, dalpha]; % first order derivative 

end

