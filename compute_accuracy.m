function accuracy = compute_accuracy(NcutDiscrete, idx)
% July 2014
% This matlab code computes accuracy of subspace clustering.
%
%
%
% Reference: Hongyang Zhang, Zhouchen Lin, Chao Zhang, Junbin Gao.
%            Robust Latent Low Rank Representation for Subspace Clustering.
%
% Copyright: Key Laboratory of Machine Perception (MOE), Peking University, Beijing



[m, n] = size(NcutDiscrete);
result = zeros(m, 1);
accuracy = 0;

ranking = perms(1:n);

for i = 1:length(ranking)
    tmp = NcutDiscrete(:, ranking(i, :));
    for j = 1:n
        result(find(tmp(:, j))) = j;
    end
    curr_accuracy = length(find(result == idx))/m;
    if curr_accuracy > accuracy
        accuracy = curr_accuracy;
    end
end