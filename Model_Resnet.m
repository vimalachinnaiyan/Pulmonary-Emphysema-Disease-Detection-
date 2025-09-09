function [Eval, pred] = Model_Resnet()

global  Train_Data Test_Data Train_Target Test_Target
% Load the pretrained ResNet-50 model
net = resnet50('Weights','none');

% Configure the training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', Test_Data, ...
    'ValidationFrequency', 5, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the ResNet-50 model
trainedNet = trainNetwork(Train_Data, net, options);

% Classify the test set
pred = classify(trainedNet, Train_Target);
act = Test_Target
pred(YPred < 0.5) = 0;
pred(YPred >= 0.5) = 1;
pred  = reshape(pred, length(pred), 1);
Eval = evaluation({pred},{act});
end