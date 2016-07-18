function [ acc, llh] = bayeslr_evaluation(theta, X_test, y_test )
% calculate the prediction error and log-likelihood
% theta: M * d, logistic regression weights
% X_test:  N0 * d, input data
% y_test:  N0 * 1, contains the label (+1/-1)

theta = theta(:,1:end-1); % only need w to evaluate accuracy and likelihood

M = size(theta, 1);  % number of particles
n_test = length(y_test); % number of evaluation data points

prob = zeros(n_test, M);
for t = 1:M
      prob(:, t) = ones(n_test,1) ./ (1 + exp( y_test.* sum(-repmat(theta(t,:), n_test, 1) .* X_test, 2)));
end
prob = mean(prob, 2);
acc = mean(prob > 0.5);
llh = mean(log(prob));

end

