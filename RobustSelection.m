function [Z, w] = RobustSelection(V)
% July 2014
% This matlab code uses ADM to solve the optimization problem:
%        min_{Z,W} ||Z||_1,  s.t.   Z = VWV',   W is diagonal,
%                               0 <= diag(W) <= 1,  trace(W) = 1.
%
%
% Reference: Hongyang Zhang, Zhouchen Lin, Chao Zhang, Junbin Gao.
%            Robust Latent Low Rank Representation for Subspace Clustering.
%
% Copyright: Key Laboratory of Machine Perception (MOE), Peking University, Beijing


[n, r] = size(V);

temp = zeros(r, 1);

%% Initialize optimization variables
w_hat = ones(r, 1)/r;
c_hat = ones(r, 1)/r;
Y_hat = zeros(n, n);
alpha_hat = zeros(r, 1);
S = V*diag(w_hat)*V';

%% Parameters setting
mu = 1e-3;
max_mu = 1e10;
rho = 1.25;
max_iter = 1000;


%% Start main loop
for iter = 1:max_iter
    
    %% update Z_hat
    tmp = S+Y_hat/mu;
    Z_hat = max(tmp-1/mu, 0);
    Z_hat = Z_hat+min(tmp+1/mu, 0);
    
    %% update w_hat
    beta = 0;
    for i = 1:r
        temp(i) = V(:, i)'*Z_hat*V(:, i)+c_hat(i)-(V(:, i)'*Y_hat*V(:, i)+alpha_hat(i))/mu;
        beta = beta+temp(i);
    end
    beta = mu*(beta-2)/r;
    for i = 1:r
        w_hat(i) = (temp(i)-beta/mu)/2;
    end
    
    %% update c_hat
    c_hat = w_hat+alpha_hat/mu;
    c_hat(c_hat < 0) = 0;
    
    %% update the multiplier Y_hat
    S = V*diag(w_hat)*V';
    Y_hat = Y_hat+mu*(S-Z_hat);
    
    %% update the multiplier alpha_hat
    alpha_hat = alpha_hat+mu*(w_hat-c_hat);
    
    %% update mu
    mu = min(max_mu, rho*mu);
    
end

Z = Z_hat;
w = w_hat;