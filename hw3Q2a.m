clear; clc; close all; 
load MNIST_database.mat;
% tmp=reshape(train_data(:,7),28,28);
% imshow(double(tmp));

% process trainset
myTrainx = train_data;
TrN = size(myTrainx, 2);
TrLabel = zeros(1,TrN); 
TrLabel(train_classlabel == 7 | train_classlabel == 1) = 1;

% process testset
myTestx = test_data;
TeN = size(myTestx, 2);
TeLabel = zeros(1, TeN);
TeLabel(test_classlabel == 7 | test_classlabel == 1) = 1;
variance = 2 * 100 * 100;

phi_train = zeros(TrN, TrN);
for i = 1 : TrN
    for j = 1 : TrN
        phi_train(j, i) = exp(-sum((myTrainx(:, i) - myTrainx(:, j)) .^ 2) / variance);
    end
end


for lambda = [0 0.001 0.1 1 5 10 20]
    if lambda == 0
        weights = TrLabel / phi_train;
    else
        I = eye(TrN);
        weights = (phi_train * phi_train' + lambda * I) \ (phi_train * TrLabel');
        weights = weights';
    end
    TrPred = weights * phi_train';
    
    phi_test = zeros(TeN, TrN);
    for i = 1 : TeN
        for j = 1 : TrN
            phi_test(i, j) = exp(-sum((myTestx(:, i) - myTrainx(:, j)) .^ 2) / variance);
        end
    end
    TePred = weights * phi_test';
    
    TrAcc = zeros(1,1000);
    TeAcc = zeros(1,1000);
    thr = zeros(1,1000);
    for i = 1:1000
        t = (max(TrPred)-min(TrPred)) * (i-1)/1000 + min(TrPred);
        thr(i) = t;
        TrAcc(i) = (sum(TrLabel(TrPred<t)==0) + sum(TrLabel(TrPred>=t)==1)) / TrN;
        TeAcc(i) = (sum(TeLabel(TePred<t)==0) + sum(TeLabel(TePred>=t)==1)) / TeN;
    end
    
    figure
    plot(thr,TrAcc,'.- ',thr,TeAcc,'-');legend({'tr','te'},'Location','southwest');
    xlim([-0.5,1.2]);
    ylim([0,1]);
    xlabel('Threshold');
    ylabel('Accuracy');
    if lambda == 0
        title('Exact Interpolation Method')
    else
        title(['Regularization(lambda = ', num2str(lambda), ')'])
        
    end
    
end
