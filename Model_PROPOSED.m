function [Eval, YPred] = Model_PROPOSED()

global Bestsol Train_Data Test_Data Train_Target Test_Target

% Define DenseNet layers
lgraph = densenetLayers([32 32 3], 10, 'NumClasses', 2);

% Define training options
options = trainingOptions('adam', ...
    'MaxEpochs', Bestsol(2), ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', Bestsol(3), ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(Train_Data, Train_Target, lgraph, options);

% Evaluate the model
YPred = classify(net, Test_Data);

actual =Test_Target;
Eval = evaluation({actual}, {YPred});

end
