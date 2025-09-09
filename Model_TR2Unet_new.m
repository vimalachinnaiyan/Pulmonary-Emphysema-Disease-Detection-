function [pxdsResults] = Model_TR2Unet_new()

global Original Feat Targets prep
% Define U-Net with residual connections and LSTM layers
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

% Check if lstmLayer is available
if exist('lstmLayer', 'file') == 2
    lstmLayerObj = [
        sequenceInputLayer(64*64*64, 'Name', 'lstm_input')
        lstmLayer(64, 'Name', 'lstm1')
        fullyConnectedLayer(64*64*64, 'Name', 'fc_lstm')
        ];
else
    error('LSTM layer not found. Ensure that the Deep Learning Toolbox is installed and up-to-date.');
end

% Placeholder for transformerLayer definition
% Commented out if not defined
% transformerLayer = customTransformerLayer(64, 'Name', 'transformer1');

lgraph = layerGraph(encoderLayers);
lgraph = addLayers(lgraph, decoderLayers);
lgraph = addLayers(lgraph, lstmLayerObj);
% lgraph = addLayers(lgraph, transformerLayer);

% Adding connections and skip connections
skipLayer = additionLayer(2, 'Name', 'skip1');
lgraph = addLayers(lgraph, skipLayer);
lgraph = connectLayers(lgraph, 'relu1_2', 'skip1/in1');
lgraph = connectLayers(lgraph, 'relu2_1', 'skip1/in2');

% Merge skip connection with decoder output
mergeLayer = additionLayer(2, 'Name', 'merge');
lgraph = addLayers(lgraph, mergeLayer);
lgraph = connectLayers(lgraph, 'skip1', 'merge/in1');
lgraph = connectLayers(lgraph, 'relu_upconv1', 'merge/in2');
lgraph = connectLayers(lgraph, 'merge', 'upconv1');

% Convert LayerGraph to DAGNetwork
net = assembleNetwork(lgraph);

% Plot the network
plot(net);

% Segmented Outcome
pxdsResults = semanticseg(prep, net, 'MiniBatchSize', 16, 'Verbose', false);
% Performance matrices
metrics = evaluation_pre(pxdsResults, prep);

end
