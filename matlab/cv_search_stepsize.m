function master_stepsize_star = cv_search_stepsize(X, y, theta0, dlog_p)
%%%%%%%%%%%%%
% In practice, we need to tune the general learning rate for adagrad.
% we exhaustive search over specified parameter values for VGD.
%%%%%%%%%%%%%

% adagrad master stepsize
master_stepsize_grid = [1e0, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6];

max_iter = 1000;

% randomly partition 20% dataset for validation
validation_ratio = 0.2; N = size(X,1);
validation_idx = randperm(N, round(validation_ratio*N));  train_idx = setdiff(1:N, validation_idx);

X_train = X(train_idx, :); y_train = y(train_idx);
X_validation = X(validation_idx, :); y_validation = y(validation_idx);

best_acc = 0; master_stepsize_star = 0.1;

dlog_p_cross_validation = @(theta) dlog_p(theta, X_train, y_train);

% grid parameters search strategy
for master_stepsize = master_stepsize_grid
    theta = vgd(theta0, dlog_p_cross_validation, max_iter, master_stepsize);
    [acc, ~] = bayeslr_evaluation(theta, X_validation, y_validation);
    if acc > best_acc
        best_acc = acc;
        master_stepsize_star = master_stepsize;
    end
    fprintf('master_stepsize = %f, current acc = %f, best acc = %f, best master_stepsize %f \n', master_stepsize, acc, best_acc, master_stepsize_star);
end

end



