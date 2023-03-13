%% Function to Load MNIST Data

function [train_data, TrLabel, test_data, TeLabel] = loadmnist
% Load the Training and Testing Data
load MNIST_database.mat train_data train_classlabel test_data test_classlabel;

% train_data       -> training data, 784x1000 matrix
% train_classlabel -> the labels of the training data, 1x1000 vector 
% test_data        -> test data, 784x250 matrix
% train_classlabel -> the labels of the test data, 1x250 vector

% Uncomment these two lines to display the image
%  tmp = reshape(train_data(:,7),28,28);
%  imshow(tmp)

% Change label to 0 & 1 for training and testing set
train_data = double(train_data);
test_data = double(test_data);
TrLabel = double(train_classlabel);
TeLabel = double(test_classlabel);

for i = 1:length(TrLabel)
   if (TrLabel(i) == 6 || TrLabel(i) == 9)
       TrLabel(i) = 1;
   else
       TrLabel(i) = 0;
   end
end

for i = 1:length(TeLabel)
   if (TeLabel(i) == 6 || TeLabel(i) == 9)
       TeLabel(i) = 1;
   else
       TeLabel(i) = 0;
   end
end
end

