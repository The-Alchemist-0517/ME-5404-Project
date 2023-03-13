clc;clear;close all;
xtrain=-1:0.05:1;
xtest=-1:0.01:1;
x0=-2:0.01:2;
ytrain=1.2*sin(pi*xtrain)-cos(2.4*pi*xtrain);
ytest=1.2*sin(pi*xtest)-cos(2.4*pi*xtest);
y0=1.2*sin(pi*x0)-cos(2.4*pi*x0);
n=[1:10,20,50,100]; % different structure of layers
% n=10;
epoches=1000;


% [mynet]=train_seq(n,xtrain,ytrain,epoches)

% % Compute Outputs of MLP when x=-3 and 3
% t_pos = 1.2*sin(pi*3) - cos(2.4*pi*3);
% t_neg = 1.2*sin(pi*-3) - cos(2.4*pi*-3);
% 
% pred_pos = mynet(3);
% pred_neg = mynet(-3);
% 
% fprintf('The output value from function for x = 3 : %.4f\n', t_pos);
% fprintf('The output value from function for x = -3: %.4f\n', t_neg);
% fprintf('The output value from approx function for x = 3  : %.4f\n', pred_pos);
% fprintf('The output value from approx function for x = -3 : %.4f\n', pred_neg);

for i=n
    [mynet]=train_seq(i,xtrain,ytrain,epoches);
    ypred=mynet(xtest);
    figure
%     x0=1:1:epoches
%     plot(x0,error,'LineWidth',2)
    hold on 
    plot(x0,y0,'LineWidth',2)
    hold on
    plot(xtrain,ytrain,'o')
    hold on
    plot(xtest,ypred,'LineWidth',2)
    ylim([-2.2 2.5])
    xlim([-2 2])
    legend('Original function','Train Points','MLP result')
    title(['Sequentional Mode(Epochs: ',num2str(epoches),' Hidden neurons: ',num2str(i),')'])
    ylabel('y-value')
    xlabel('x-value')
    hold on
    
end

% according to the provided code
function [net] = train_seq(n,x,y,epochs)

% Construct a 1-n-1 MLP and conduct sequential training.
train_num=size(x,2);
error=zeros(epochs,1);

% 1. Change the input to cell array form for sequential training
x_c = num2cell(x, 1);
y_c = num2cell(y, 1);

% 2. Construct and configure the MLP
net = fitnet(n,'trainlm');
% net.divideFcn = 'dividetrain'; % input for training only
% net.divideParam.trainRatio=1.0;
% net.divideParam.valRatio=0.0; 
% net.divideParam.testRatio=0.0; 
net.trainParam.lr = 0.1;
% net.trainParam.epochs = epochs;

% 3. Train the network in sequential mode
for i = 1 : epochs
    idx = randperm(train_num); % shuffle the input
    net = adapt(net, x_c(:,idx), y_c(:,idx));
    mat_perf_train=perform(net,y_c,net(x_c));
    display(['Neurons: ',num2str(n)])
    display(['Performance: ',num2str(mat_perf_train)])
    error(i,1)= mat_perf_train; 
end
figure
x0=1:1:epochs;
plot(x0,error,'LineWidth',2)
title(['Loss Function(Epochs: ',num2str(epochs),' Hidden neurons: ',num2str(n),')'])
xlabel('iterations')
ylabel('Loss')

end
