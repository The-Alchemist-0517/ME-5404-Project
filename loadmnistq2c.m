%% Function to Load MNIST Data

function loadmnistq2c
% Load the Training and Testing Data
load MNIST_database.mat train_data train_classlabel;

% Plots of Training Image Centers
trainIdx_0 = find(train_classlabel==0);
trainIdx_1 = find(train_classlabel==1);
trainIdx_2 = find(train_classlabel==2);
trainIdx_3 = find(train_classlabel==3);
trainIdx_4 = find(train_classlabel==4);
trainIdx_5 = find(train_classlabel==5);
trainIdx_6 = find(train_classlabel==6);
trainIdx_7 = find(train_classlabel==7);
trainIdx_8 = find(train_classlabel==8);
trainIdx_9 = find(train_classlabel==9);

train_data_0 = train_data(:,trainIdx_0);
train_data_1 = train_data(:,trainIdx_1);
train_data_2 = train_data(:,trainIdx_2);
train_data_3 = train_data(:,trainIdx_3);
train_data_4 = train_data(:,trainIdx_4);
train_data_5 = train_data(:,trainIdx_5);
train_data_6 = train_data(:,trainIdx_6);
train_data_7 = train_data(:,trainIdx_7);
train_data_8 = train_data(:,trainIdx_8);
train_data_9 = train_data(:,trainIdx_9);

trdatacenter_0 = mean(train_data_0, 2);
trdatacenter_1 = mean(train_data_1, 2);
trdatacenter_2 = mean(train_data_2, 2);
trdatacenter_3 = mean(train_data_3, 2);
trdatacenter_4 = mean(train_data_4, 2);
trdatacenter_5 = mean(train_data_5, 2);
trdatacenter_6 = mean(train_data_6, 2);
trdatacenter_7 = mean(train_data_7, 2);
trdatacenter_8 = mean(train_data_8, 2);
trdatacenter_9 = mean(train_data_9, 2);

figure;
subplot(2, 5, 1);
imshow(reshape(trdatacenter_0,28,28));
title('Class0');
subplot(2, 5, 2);
imshow(reshape(trdatacenter_1,28,28));
title('Class1');
subplot(2, 5, 3);
imshow(reshape(trdatacenter_2,28,28));
title('Class2');
subplot(2, 5, 4);
imshow(reshape(trdatacenter_3,28,28));
title('Class3');
subplot(2, 5, 5);
imshow(reshape(trdatacenter_4,28,28));
title('Class4');
subplot(2, 5, 6);
imshow(reshape(trdatacenter_5,28,28));
title('Class5');
subplot(2, 5, 7);
imshow(reshape(trdatacenter_6,28,28));
title('Class6');
subplot(2, 5, 8);
imshow(reshape(trdatacenter_7,28,28));
title('Class7');
subplot(2, 5, 9);
imshow(reshape(trdatacenter_8,28,28));
title('Class8');
subplot(2, 5, 10);
imshow(reshape(trdatacenter_9,28,28));
title('Class9');

end