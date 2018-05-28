function [U_hat Sigma_hat V_hat A_hat E_hat iter] = inexact_alm_rpca(D, lambda, tol, max_iter, rho)

% Oct 2009
% This matlab code implements the inexact augmented Lagrange multiplier 
% method for Robust PCA.
%
% D - m x n matrix of observations/data (required input)
%
% lambda - weight on sparse error term in the cost function
%
% tol - tolerance for stopping criterion.
%     - DEFAULT 1e-7 if omitted or -1.
%
% max_iter - maximum number of iterations
%         - DEFAULT 1000, if omitted or -1.
% 
% rho - increasing ratio of mu
%         - DEFAULT 1.7, if omitted or -1.
%
% Initialize A,E,Y,u
% while ~converged 
%   minimize (inexactly, update A and E only once)
%     L(A,E,Y,u) = |A|_* + lambda * |E|_1 + <Y,D-A-E> + mu/2 * |D-A-E|_F^2;
%   Y = Y + \mu * (D - A - E);
%   update \mu; %In this version, the next \mu is predicted and clipped to [\mu,\rho*\mu]
% end
%
% The speed can be slightly improved by using Y to represent Y/mu. Then
% some scaling of matrices can be avoided. This version of inexact ALM for Robust
% PCA realizes this idea. -- Zhouchen Lin, Dec. 29, 2009.
%
% Minming Chen (cmmfir@gmail.com);
% Zhouchen Lin (zhoulin@microsoft.com; zhouchenlin@gmail.com)
%
% Reference: Zhouchen Lin, Minming Chen, and Yi Ma, The Augmented Lagrange Multiplier Method 
%for Exact Recovery of Corrupted Low-Rank Matrix, http://perception.csl.illinois.edu/matrix-rank/Files/Lin09-MP.pdf
%
% Copyright: Microsoft Research Asia, Beijing

addpath inexact_alm_rpca/PROPACK;
addpath PROPACK;

[m n] = size(D);
n1 = max([m n]);
n2 = min([m n]);
% 
if nargin < 2
    lambda = 2 / sqrt(n1);
elseif lambda == -1
    lambda = 2 / sqrt(n1);
end
%lambda = 5/ sqrt(m*n);%k/sqrt(m*n)
if nargin < 3
    tol = 1e-2;
elseif tol == -1
    tol = 1e-2;
end

if nargin < 4
    max_iter = 1000;
elseif max_iter == -1
    max_iter = 1000;
end

if nargin < 5
    rho = 1.7;
elseif rho == -1
    rho = 1.7;
end

%% initialize
Y = D;
norm_two = lansvd(Y, 1, 'L');
norm_inf = norm( Y(:), inf) / lambda;
norm_dual = max(norm_two, norm_inf);
Y = Y / norm_dual;

A_hat = zeros(m, n);
E_hat = zeros(m, n);
mu = 20 / norm_two; % this one can be tuned
mu_bar = mu * 1e9;
tol2 = 1e-5;
norm_D = norm(D, 'fro');

Y = Y / mu; % Y now actually stands for Y/mu in the original implementation

iter = 0;
converged1 = false;
converged2 = false;
stopCriterion1 = 1;
stopCriterion2 = 1;
sv = min([10, n2]);

%% IALM
while ~converged1 || ~converged2
    iter = iter + 1;
    
    %% update E
    temp = D - A_hat + Y;
    E_hat = max(temp - lambda/mu, 0);
    E_hat = E_hat + min(temp + lambda/mu, 0);

    %% update A
    if choosvd(n2, sv) == 1
        [U S V] = lansvd(D - E_hat + Y, sv, 'L');
    else
        [U S V] = svd(D - E_hat + Y, 'econ');
    end
    diagS = diag(S);
    svp = length(find(diagS > 1/mu));
    if svp < sv
        sv = min([svp + 1, n2]);
    else
        sv = min([svp + round(0.05*n2), n2]);
    end
    temp = A_hat;
    A_hat = U(:, 1:svp) * diag(diagS(1:svp) - 1/mu) * V(:, 1:svp)';
    
    %% one stop criterion is mu_k*||A_{k+1}-A_k||/||D|| < tol2
    converged2 = false;
    stopCriterion2 = mu*norm(A_hat - temp, 'fro') / norm_D;
    if stopCriterion2 < tol2
        converged2 = true;
    end      
    
    %% the other stop criterion is ||D-A-E||/||D|| < tol  
    converged1 = false;
    temp = D - A_hat - E_hat;      
    stopCriterion1 = norm(temp, 'fro') / norm_D;
    if stopCriterion1 < tol
        converged1 = true;
    end    
    
    %% update mu
    mu_old = mu;
    if converged2
       mu = min(mu*rho, mu_bar);
    end
    
    %% update Y
    Y = Y + temp;    
    if mu_old ~= mu
       Y = Y*(mu_old/mu);
    end
        
    norm_Y = mu*norm(Y(:),inf)/lambda;
    if mod(iter, 20) == 0
        disp(['#svd ' num2str(iter) ' r(A) ' num2str(svp)...
            ' |E|_0 ' num2str(length(find(abs(E_hat)>0)))...
            ' stopCriterion1 ' num2str(stopCriterion1)...
            ' stopCriterion2 ' num2str(stopCriterion2)...
            ' ||Y|| ' num2str(norm_Y)...
            ' mu ' num2str(mu)]);
    end    
    
    if (~converged1 || ~converged2) && iter >= max_iter
        disp('Maximum iterations reached') ;
        converged1 = 1;       
        converged2 = 1;       
    end
end

U_hat = U(:, 1:svp);
Sigma_hat = diag(diagS(1:svp) - 1/mu);
V_hat = V(:, 1:svp);
