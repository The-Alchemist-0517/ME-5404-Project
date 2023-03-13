clear; clc; close all;
load('MNIST_database.mat');
% generate the training data
trIdx = find(train_classlabel == 0 | train_classlabel == 2 | train_classlabel == 3 | train_classlabel == 4| train_classlabel == 5 | train_classlabel == 6 | train_classlabel == 8 | train_classlabel == 9);
myTrain = train_data(:, trIdx);
TrLabel = train_classlabel(trIdx);
Train_data = size(myTrain, 2);%number of columns of mytrain

% generate the test data
teIdx = find(test_classlabel == 0 | test_classlabel == 2 | test_classlabel == 3 | test_classlabel == 4 | test_classlabel == 5 | test_classlabel == 6 | test_classlabel == 8 | test_classlabel == 9);
myTest = test_data(:, teIdx);
TeLabel = test_classlabel(teIdx);
Test_data = size(myTest, 2);

%construct the net
nRows = 10;
nCols = 10;
numNeurons = nRows*nCols;
weights = rand(28*28,numNeurons);

%construct training parameter
% numIteration = 500;
numIteration = 1000;
% record = 1000;
record = [0,10,20,50,100:100:numIteration];
r = 1;iter=0;
learning_rate0 = 0.1;
sigma0 = sqrt(nRows^2+nCols^2)/2;
tau0 = numIteration/log(sigma0);
TeAcc = zeros(1, size(record, 2)); 
TrAcc = zeros(1, size(record, 2));

%training process
while iter <= numIteration
    sampleIdx = randi(Train_data);
    sample = myTrain(:,sampleIdx);
    distance = sum((sample-weights).^2);
    [~,minIdx] = min(distance);
    [row, col] = ind2sub([nRows,nCols],minIdx);
    sigma = sigma0*exp(-iter/tau0);
    learning_rate = learning_rate0*exp(iter/numIteration);
    for i = 1:numNeurons
        [neuronRow, neuronCol] = ind2sub([nRows, nCols],i);
        distance2neuron = sqrt((neuronRow-row)^2+(neuronCol-col)^2);
        h = exp(-distance2neuron^2/(2*sigma)^2);
        weights(:,i) = weights(:,i)+h*learning_rate*(sample-weights(:,i));
    end 
    
    %calculate neurons'label
     if iter == record(r)
        map = zeros(10,numNeurons);
        for i = 1 : Train_data
            sample = myTrain(:,i);%same numbers of rows as weights
            [~,minIdx] = min(sum((sample-weights).^2));%According to the struct of sample here,the minIdx is the column number
            map(TrLabel(i)+1,minIdx) = map(TrLabel(i)+1,minIdx)+1;%find the closest one and +1 
        end 
        
        neurons_label = zeros(1,numNeurons);
        neurons_value = zeros(1,numNeurons);
        for i = 1 : numNeurons
            [val, idx] = max(map(:,i));%find the row of the max  value, indicates the label
            neurons_label(i) = idx - 1;
            neurons_value(i) = val;
        end
        
        %calculate test & train accuracy
        for i = 1 : Test_data
            sample = myTest(:,i);
            [~,minIdx] = min(sum((sample-weights).^2));
            TeAcc(r) = TeAcc(r) + (neurons_label(minIdx) == TeLabel(i));    
        end
        for i = 1 : Train_data
            sample = myTrain(:,i);
            [~,minIdx] = min(sum((sample-weights).^2));
            TrAcc(r) = TrAcc(r) + (neurons_label(minIdx) == TrLabel(i));    
        end
        TeAcc(r) = TeAcc(r) / Test_data;
        TrAcc(r) = TrAcc(r) / Train_data;
        r = r + 1;
    end
    iter = iter+1;    
end

%weights visualize
trained_weights = [];
for i = 0 : 9
    weights_row = [];
    for j = 1 : 10
        weights_row = [weights_row, reshape(weights(:, i*10+j), 28, 28)];
    end
    trained_weights = [trained_weights; weights_row];
end
figure
imshow(imresize(trained_weights, 4))
title('weights visualization')

% show the conceptual map
neurons_label = reshape(neurons_label, 10, 10)';
neurons_value = neurons_value/max(neurons_value);
neurons_value = reshape(neurons_value, [10,10])';
figure
img = imagesc(neurons_label);
img.AlphaData = neurons_value;
for i = [0,2, 3,4,5,6,8,9]
    neurons_label(neurons_label == i) = num2str(i);
end
label = num2str(neurons_label, '%s');       
[x, y] = meshgrid(1:10);  
hStrings = text(x(:), y(:), label(:), 'HorizontalAlignment', 'center');
title('conceptual map')

% show the accuracy
figure
hold on
plot(record, TrAcc, 'linewidth', 2)
plot(record, TeAcc, 'linewidth', 2)
hold off
legend('Train Accuracy','Test Accuracy', 'Location', 'Best')
title('accuracy according to iteration')

    





