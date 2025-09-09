function [Eval, YTest] = Model_CNN()

global Train_Data Test_Data Train_Target Test_Target

size_of_one_dim = 128;
Train_data = zeros(size_of_one_dim, size_of_one_dim, 1);
for i = 1:size(Train_Data, 1)
    i
    tr_data = imresize(Train_Data(i, :), [1 size_of_one_dim * size_of_one_dim]);
    Train_data( :,:, :, i) = reshape(tr_data, [size_of_one_dim size_of_one_dim 1]);
end

Test_data = zeros(size_of_one_dim, size_of_one_dim, 1);
for i = 1:size(Test_Data, 1)
    
    i
    test_data = imresize(Test_Data(i, :), [1 size_of_one_dim * size_of_one_dim]);
    Test_data(:, :, :, i) = reshape(test_data, [size_of_one_dim size_of_one_dim 1]);
end

layers = [
    imageInputLayer([128 128 1]) % 22X1X1 refers to number of features per sample
    convolution2dLayer(20, 5)  %3,16,'Padding','same')
    convolution2dLayer(20, 5)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(size(Train_Target, 2))
    regressionLayer];

options = trainingOptions('sgdm','MaxEpochs',5, ...
    'Verbose', false,...
    'Plots', 'training-progress', ...
    'InitialLearnRate', 0.01);

rng('default')
Y = double(Train_Target);
Test_Target = double(Test_Target);
[net, info] = trainNetwork(Train_data, Y, layers, options);
YTest = predict(net, Test_data);
YTest(YTest < 0.5) = 0;
YTest(YTest >= 0.5) = 1;
Eval = evaluation({Test_Target}, {YTest});
end