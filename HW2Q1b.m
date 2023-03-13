clc;clear;close all;
threshold=1e-8;
x(1)=rand/2; % initialize x between(-1,1)
y(1)=rand/2; % initialize y between(-1,1)
fxy(1)=(1-x(1))^2+100*(y(1)-x(1)^2)^2; % initial f(x,y)
iter=1;

while fxy(iter)>threshold
    % calculate g-matrix
    g=[2*(x(iter)-1)+400*x(iter)*(x(iter)*x(iter)-y(iter));200*(y(iter)-x(iter)*x(iter))];
    % calculate h-matrix
    h=[2+1200*x(iter)*x(iter)-400*y(iter), -400*x(iter);-400*x(iter),200];
    res=([x(iter);y(iter)]-h^(-1)*g)';
    iter=iter+1;
    % update x,y
    x(iter)=res(1);
    y(iter)=res(2);
    % update f(x,y)
    fxy(iter)=(1-x(iter))^2+100*(y(iter)-x(iter)^2)^2;
end
t=1:iter;
figure

% subplot(121)
plot(x,y,'linewidth',2)
grid on
% hold on
% plot(t,y,'linewidth',2)
title(['x-value, y-value'])
% legend('x-value', 'y-value')
xlabel('x')
ylabel('y')

figure
% subplot(122)
plot(t,fxy,'linewidth',2)
grid on
title(['Newton method'])
legend('f(x,y)-value')
xlabel('iteration')

% sgtitle('Newton method')
