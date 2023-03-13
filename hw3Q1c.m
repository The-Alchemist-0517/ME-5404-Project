%% Homework 3 Q1c

clc;
clear;

% Set Random Seed
rng(42);

% Standard Deviation
std = 0.1;

% Regularization Factor
lambda = 20; 

% Training and Testing Data Set
x_train = [-1:0.05:1];
x_test  = [-1:0.01:1];
m_train = size(x_train,2);
y_train = 1.2 * sin(pi * x_train) - cos(2.4 * pi * x_train) + 0.3 * randn(1,m_train);
y_test  = 1.2 * sin(pi * x_test) - cos(2.4 * pi * x_test);

% Gaussian Functions for Training Set
r_train   = dist(x_train);
phi_train = exp(-r_train.^2 / (2 * std^2));  
w = pinv((phi_train.' * phi_train) + lambda * eye(m_train)) * phi_train.' * y_train.';
y_pred_train = (phi_train.' * w).';

% Gaussian Functions for Testing Set
r_test   = dist(x_train.',x_test);
phi_test = exp(-r_test.^2 / (2 * std^2)); 
y_pred_test = (phi_test.' * w).';

% MSE of this Function Approximation
err_test = mse(y_pred_test, y_test);
err_train = mse(y_pred_train, y_test);
% Plot of Trained Output vs Desired Output
figure(1);
hold on;
title(['Training Results for Lambda =' num2str(lambda)]);
txt=['Mean Square Error(test) : ' num2str(err_test)];
text(-0.95,1.4,txt);
txt=['Mean Square Error(train) : ' num2str(err_train)];
text(-0.95,1.2,txt);
plot(x_test, y_test, 'LineWidth', 2);
plot(x_test, y_pred_test, 'LineWidth', 2);
plot(x_train, y_train, 'ok', 'LineWidth', 1);
legend({'Desired Output Function','Estimated Output Function','Training Data'},'Location','northwest');
