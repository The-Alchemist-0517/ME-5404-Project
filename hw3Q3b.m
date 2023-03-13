
% generate the training data
trainX = rands(2,500); % 2x500 matrix, column-wise points
plot(trainX(1,:),trainX(2,:),'+r')
data = size(trainX,2);

%construct the net
nRows = 20;
nCols = 20;
numNeurons = nRows*nCols;
weights = rand(2,numNeurons);

%construct training parameter
numIteration = 500;
% numIteration = 1000;
% record = 1000;
record = [0,10,20,50,100:100:numIteration];
r = 1;i=0;
learning_rate0 = 0.1;
sigma0 = sqrt(nRows^2+nCols^2)/2;
tau0 = numIteration/log(sigma0);

%training process
while i <= numIteration
    sampleIdx = randi(data);
    sample = trainX(:,sampleIdx);
    distance = sum((sample-weights).^2);
    [minDist,minIdx] = min(distance);
    [row, col] = ind2sub([nRows,nCols],minIdx);
    sigma = sigma0*exp(-i/tau0);
    learning_rate = learning_rate0*exp(i/numIteration);
    for j = 1:numNeurons
        [neuronRow, neuronCol] = ind2sub([nRows, nCols],j);
        distance2neuron = sqrt((neuronRow-row)^2+(neuronCol-col)^2);
        h = exp(-distance2neuron^2/(2*sigma)^2);
        weights(:,j) = weights(:,j)+h*learning_rate*(sample-weights(:,j));
    end 
    
    %plot
     if i == record(r)
        figure
        hold on
        plot(trainX(1,:),trainX(2,:),'+r');
        for k = 1 : 10
            plot(weights(1, k*20-19:k*20), weights(2, k*20-19:k*20), '+b-');
            plot(weights(1, k:20:end), weights(2, k:20:end), '+b-');
        end
        hold off
        title(['2-dim SOM result(iteration=', num2str(i), ')']);
        axis equal
        r = r + 1;
    end
    i = i+1;    
end
    




