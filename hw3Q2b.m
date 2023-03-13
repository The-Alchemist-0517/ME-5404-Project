%% Homework 3 Q2b

clc;
clear;

% Set Random Seed
rng(42);

% Load the Training and Testing Data
[TrData, TrLabel, TeData, TeLabel] = loadmnist;

sample = size(TrData, 2);

% Parameters for RBF Fix Center
center_idx  = randsample(sample,100);
TrData_100  = TrData(:, center_idx);
d_max_train = max(max(dist(TrData_100)));
sigma_cal   = d_max_train / sqrt(2 * 100);

for sigma = [sigma_cal 0.1 1 1 10 100 1000 10000]

    phi_train = exp(-(dist(TrData.',TrData_100).^2) / (2 * sigma^2));
    w = pinv((phi_train.' * phi_train)) * phi_train.' * TrLabel.';
    
    phi_test  = exp(-(dist(TeData.',TrData_100).^2) / (2 * sigma^2));
    
    TrPred = phi_train * w;
    TePred = phi_test * w;
    
    % Evaluate Performance of RBFN
    TrAcc = zeros(1,1000);
    TeAcc = zeros(1,1000);
    thr   = zeros(1,1000);
    TrN   = length(TrLabel);
    TeN   = length(TeLabel);
    for i = 1:1000
        t = (max(TrPred)-min(TrPred)) * (i-1)/1000 + min(TrPred); 
        thr(i) = t;
        TrAcc(i) = (sum(TrLabel(TrPred<t)==0) + sum(TrLabel(TrPred>=t)==1)) / TrN;
        TeAcc(i) = (sum(TeLabel(TePred<t)==0) + sum(TeLabel(TePred>=t)==1)) / TeN;
    end
    
    figure;
    plot(thr,TrAcc,'.- ',thr,TeAcc,'-');
    legend('tr','te');
    xlabel('Threshold');
    ylabel('Accuracy');
    xlim([-0.4 1.4])
    ylim([0 1])
    title(['Fixed Centers Selected at Random(width =', num2str(sigma), ')'])
    
end