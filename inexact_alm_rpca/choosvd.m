function y = choosvd( n, d)
%This function determines when to use full or partial SVD to compute the largest d singular vectors
%n is the smaller of the numbers of columns or rows of a matrix.
%y==1 means to use partial sVD
%y==0 means to use full SVD
%
% Minming Chen (cmmfir@gmail.com);
% Zhouchen Lin (zhoulin@microsoft.com; zhouchenlin@gmail.com)
%
% Reference: Zhouchen Lin, Minming Chen, and Yi Ma, The Augmented Lagrange Multiplier Method 
%for Exact Recovery of Corrupted Low-Rank Matrix, http://perception.csl.illinois.edu/matrix-rank/Files/Lin09-MP.pdf
%
% Copyright: Microsoft Research Asia, Beijing

if n <= 100 
    if d / n <= 0.02
        y = 1;
    else
        y = 0;
    end
elseif n <= 200
    if d / n <= 0.06
        y = 1;
    else
        y = 0;
    end
elseif n <= 300
    if d / n <= 0.26
        y = 1;
    else
        y = 0;
    end
elseif n <= 400
    if d / n <= 0.28
        y = 1;
    else
        y = 0;
    end
elseif n <= 500
    if d / n <= 0.34
        y = 1;
    else
        y = 0;
    end
else
    if d / n <= 0.38
        y = 1;
    else
        y = 0;
    end
end