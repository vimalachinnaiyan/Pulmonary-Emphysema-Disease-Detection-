function [pxdsResults] = Model_TR2Unet()

global Original Feat Targets prep
% Define U-Net with residual connections, LSTM and transformer layers
encoderLayers = [
    imageInputLayer([128 128 3], 'Name', 'input')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1_1')
    reluLayer('Name', 'relu1_1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1_2')
    reluLayer('Name', 'relu1_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    ];

decoderLayers = [
    transposedConv2dLayer(3, 64, 'Stride', 2, 'Cropping', 'same', 'Name', 'upconv1')
    reluLayer('Name', 'relu_upconv1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2_1')
    reluLayer('Name', 'relu2_1')
    ];

lstmLayer = [
    sequenceInputLayer(64*64*64, 'Name', 'lstm_input')
    lstmLayer(64, 'Name', 'lstm1')
    fullyConnectedLayer(64*64*64, 'Name', 'fc_lstm')
    ];

transformerLayer = customTransformerLayer(64, 'Name', 'transformer1');

lgraph = layerGraph(encoderLayers);
lgraph = addLayers(lgraph, decoderLayers);
lgraph = addLayers(lgraph, lstmLayer);
lgraph = addLayers(lgraph, transformerLayer);

% Adding connections and skip connections
skipLayer = additionLayer(2, 'Name', 'skip1');
lgraph = addLayers(lgraph, skipLayer);
lgraph = connectLayers(lgraph, 'relu1_2', 'skip1/in1');
lgraph = connectLayers(lgraph, 'relu2_1', 'skip1/in2');
lgraph = connectLayers(lgraph, 'skip1', 'upconv1');

res_skipLayer = additionLayer(2, 'Name', 'res_skip1');
lgraph = addLayers(lgraph, res_skipLayer);
lgraph = connectLayers(lgraph, 'conv1_2', 'res_skip1/in1');
lgraph = connectLayers(lgraph, 'relu2_1', 'res_skip1/in2');
lgraph = connectLayers(lgraph, 'res_skip1', 'upconv1');

lgraph = connectLayers(lgraph, 'relu1_2', 'lstm_input');
lgraph = connectLayers(lgraph, 'fc_lstm', 'upconv1');

lgraph = connectLayers(lgraph, 'relu1_2', 'transformer1');
lgraph = connectLayers(lgraph, 'transformer1', 'upconv1');

% Plot the network
plot(lgraph);

% Segmented Outcome
pxdsResults = semanticseg(prep, lgraph, 'MiniBatchSize',16, 'Verbose',false);
% Performance matrices
metrics = evaluation_pre(pxdsResults, prep);


end