% Minming Chen (cmmfir@gmail.com);
% Zhouchen Lin (zhoulin@microsoft.com; zhouchenlin@gmail.com)
%
% Reference: Zhouchen Lin, Minming Chen, and Yi Ma, The Augmented Lagrange Multiplier Method 
%for Exact Recovery of Corrupted Low-Rank Matrix, http://perception.csl.illinois.edu/matrix-rank/Files/Lin09-MP.pdf
%
% Copyright: Microsoft Research Asia, Beijing

newDataFlag = 1;

if newDataFlag
    
%     clear ;
%     clc ;
    close all ;

    m = 400 ; rho_r = 0.1; rho_s = 0.1;
    n = m ;
    r = round(rho_r*min(m,n)) ;
    p = rho_s ;
    U = (randn(m,r)); V = (randn(n,r));
    A = U*V' ;
    
    temp = randperm(m*n) ;
    numCorruptedEntries = round(p*m*n) ;
    corruptedPositions = temp(1:numCorruptedEntries) ;
    E = zeros(m,n) ;
    E(corruptedPositions) = 1000 * (rand(numCorruptedEntries,1)-0.5) ;
    
    D = A + E ;
            
end

lambda = 1/sqrt(m) ;

tol = 1e-7;
maxIter = -1;

tic ;
[A_dual, E_dual, numIter] = inexact_alm_rpca(D, lambda, tol, maxIter, 1.7) ;
tElapsed = toc ;
    
disp('Relative error in estimate of A') ;
error = norm(A_dual-A,'fro')/norm(A,'fro');
disp(error);
disp('Relative error in estimate of E') ;
disp(norm(E_dual-E,'fro')/norm(E,'fro')) ;
disp('Number of iterations') ;
disp(numIter) ;
disp('Rank of estimated A') ;
disp(rank(A_dual)) ;
disp('0-norm of estimated E') ;
disp(length(find(abs(E_dual)>0))) ;
disp('|D-A-E|_F');
disp(norm(D-A_dual-E_dual,'fro'));
disp('Time taken (in seconds)') ;
disp(tElapsed) ;
disp('obj value');
disp(sum(svd(A_dual))+lambda*sum(sum(abs(E_dual))));
disp('original value');
disp(sum(svd(A))+lambda*sum(sum(abs(E))));

