clear;clc;close all;
x=[0, 0.8, 1.6, 3, 4, 5;1, 1, 1, 1, 1, 1];
y=[0.5, 1, 4, 5, 6, 9];

% LLS
LLS=(x*x')^(-1)*x*y';
t=-1:0.01:7;
ft1=LLS(1)*t+LLS(2);
figure
plot(t,ft1,'LineWidth',2)
hold on
plot(x(1,:),y,'o','MarkerSize',10)
legend('LLS results','pairs')
title('LLS method')


% % LMS
rate=0.0001;epochs=100;
LMS=zeros(epochs+1,2);
LMS(1,:)=rand(1,2);

for i=1:epochs
    e=y-LMS(i,:)*x;
    LMS(i+1,:)=LMS(i,:)+rate*e*x';
end
ft2=LMS(epochs+1,1)*t+LMS(epochs+1,2);

figure
plot(t,ft2,'LineWidth',2)
hold on
plot(x(1,:),y,'o','MarkerSize',10)
legend('LMS results','pairs')
title('LMS method')

figure
t=1:101;
plot(t,LMS(:,1),'LineWidth',2)
hold on
plot(t,LMS(:,2),'LineWidth',2)
title('LMS weight trajectory(learning rate=0.0001)')
legend('w','b')
