function[Eval, pred] = Model_LSTM()

global Train_Data Test_Data Train_Target Test_Target

numFeatures = size(Train_Data, 2); % Assuming Train_Data is [features, time steps, batches]
numHiddenUnits = 10;
numResponses = size(Train_Target, 2); % Assuming Train_Target is [responses, time steps]

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(round(numHiddenUnits), 'OutputMode', 'sequence')
    fullyConnectedLayer(50)
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 5;
miniBatchSize = 20;

options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', 0.01, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'never', ...
    'Plots', 'training-progress', ...
    'Verbose', 0);
 % Example:
pred = ones(1, 650); 
Train_Data = arrayfun(@(x) rand(3, size(Train_Data, 1)), 1:50, 'UniformOutput', false); % 50 sequences, each [3 features, 100 time steps]
Train_Target = arrayfun(@(x) rand(1, size(Train_Target, 1)), 1:50, 'UniformOutput', false); % 50 sequences, each [1 response, 100 time steps]
Test_Data = arrayfun(@(x) rand(3, size(Test_Data, 1)), 1:10, 'UniformOutput', false); % 10 sequences, each [3 features, 100 time steps]
Test_Target = arrayfun(@(x) randi([0, 1], size(Test_Target, 1)), 1:10, 'UniformOutput', false); % 10 sequences, each [1 response, 100 time steps]
% Note: Ensure that Train_Data is in the correct format [features, time steps, batches]
net = trainNetwork(Train_Data, Train_Target, layers, options);
YPred = predict(net, Test_Data, 'MiniBatchSize', 1);
pred(YPred{1} < 0.5) = 0;
act = Test_Target';
actual =act{1};
Eval = evaluation({actual}, {pred});
end


