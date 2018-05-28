% July 2014
% This matlab code implements Robust Latent Low Rank Representation for
% Subspace Clustering.
%
% Synthetic test
%
% lambda - weight on sparse error term in the cost function of RPCA
%
%
% Reference: Hongyang Zhang, Zhouchen Lin, Chao Zhang, Junbin Gao.
%            Robust Latent Low Rank Representation for Subspace Clustering.
%
% Copyright: Key Laboratory of Machine Perception (MOE), Peking University, Beijing


clc;
clear;

addpath inexact_alm_rpca;
addpath Ncut_9;
addpath inexact_alm_rpca/PROPACK;

%%-------------------Data Generation-----------------------

basis1 = rand(100, 4);
basis1 = orth(basis1);
basis2 = rand(100, 4);
basis2 = orth(basis2);
basis3 = rand(100, 4);
basis3 = orth(basis3);
basis4 = rand(100, 4);
basis4 = orth(basis4);
basis5 = rand(100, 4);
basis5 = orth(basis5);

sam_num = 20;

data1 = basis1*randn(4, sam_num);
data2 = basis2*randn(4, sam_num);
data3 = basis3*randn(4, sam_num);
data4 = basis4*randn(4, sam_num);
data5 = basis5*randn(4, sam_num);

clean_data = [data1 data2 data3 data4 data5];

[m n] = size(clean_data);


accuracy1 = zeros(1, 11);
accuracy2 = zeros(1, 11);

label = [ones(sam_num, 1); 2*ones(sam_num, 1); 3*ones(sam_num, 1); 4*ones(sam_num, 1); 5*ones(sam_num, 1)];

lambda = 0.08;

for times = 1:10
    
    count = 0;

    for outlier_per = 0:0.1:1
        
        times
        outlier_per
        
        count = count+1;
        
        %% Add Gaussian noises
        outlier_position = randperm(m*n);
        outlier_position = outlier_position(1:outlier_per*m*n);
        noisy_data = clean_data;
        noisy_data(outlier_position) = noisy_data(outlier_position)+2*(rand(1, outlier_per*m*n)-0.5);  % sparse noises
        
        %% Robust LatLRR and RSI
        [~, ~, V_robust, ~, ~, ~] = inexact_alm_rpca(noisy_data, lambda); % The parameter lambda can be tuned.
        [Z, ~] = RobustSelection(V_robust);
        [~, r] = size(V_robust);
        V = V_robust;
        
        NcutDiscrete = ncutW(r*abs(Z), 5);
        accuracy2(count) = accuracy2(count)+compute_accuracy(NcutDiscrete, label); % Robust LatLRR
        NcutDiscrete = ncutW(abs(V*V'), 5);
        accuracy1(count) = accuracy1(count)+compute_accuracy(NcutDiscrete, label); % RSI
        
        
    end

end



plot(0:10:100, accuracy1*10, 'g--s', 'linewidth', 2);
hold on;
plot(0:10:100, accuracy2*10, 'r-*', 'linewidth', 2);
grid on;
legend(['RSI (\lambda=', num2str(lambda), ')'], ['Robust LatLRR (\lambda=', num2str(lambda), ')']);
xlabel('percentage of corruptions(%)', 'fontsize', 10);
ylabel('segmentation accuracy(%)', 'fontsize', 10);
