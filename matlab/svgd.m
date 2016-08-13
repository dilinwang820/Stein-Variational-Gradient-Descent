function  theta = svgd(theta0, dlog_p, max_iter, master_stepsize, h, auto_corr, method)

%%%%%%%%
% Bayesian Inference via Stein Variational Gradient Descent

% input:
%   -- theta0: initialization of particles, m * d matrix (m is the number of particles, d is the dimension)
%   -- dlog_p: function handle of first order derivative of log p(x)
%   -- max_iter: maximum iterations
%   -- master_stepsize: the general learning rate for adagrad
%   -- h/bandwidth: bandwidth for rbf kernel. Using median trick as default
%   -- auto_corr: momentum term
%   -- method: use adagrad to select the best \epsilon

% output:
%   -- theta: a set of particles that approximates p(x)
%%%%%%%%

if nargin < 4; master_stepsize = 0.1; end;

% for the following parameters, we always use the default settings
if nargin < 5; h = -1; end;
if nargin < 6; auto_corr = 0.9; end;
if nargin < 7; method = 'adagrad'; end;

switch lower(method)
    
    case 'adagrad'
        %% AdaGrad with momentum
        theta = theta0;
        
        fudge_factor = 1e-6;
        historial_grad = 0;
        
        for iter = 1:max_iter
            grad = KSD_KL_gradxy(theta, dlog_p, h);   %\Phi(theta)
            if historial_grad == 0
                historial_grad = historial_grad + grad.^2;
            else
                historial_grad = auto_corr * historial_grad + (1 - auto_corr) * grad.^2;
            end
            adj_grad = grad ./ (fudge_factor + sqrt(historial_grad));
            theta = theta + master_stepsize * adj_grad; % update
        end
        
    otherwise
        error('wrong method');
end
end
