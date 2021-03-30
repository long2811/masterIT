%% Specify the dataset directory
currentFolder = pwd;
imds = imageDatastore(currentFolder, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
%% Getting number of labels
labelCount = countEachLabel(imds);
%% Getting Image Size from first image in 'imds'
img = readimage(imds,1);
imageSize = size(img);
%% Specify the number of files used for training
filesToTrain = 100;
[imdsTrain,imdsValidation] = splitEachLabel(imds,filesToTrain,'randomize');
%% Declare the CNN network 
layers = [
    imageInputLayer(size(img))
    
    convolution2dLayer(64,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(32,'Stride',8)
    
    convolution2dLayer(32,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(16,'Stride',16)
    
    convolution2dLayer(16,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

%% Specify Training Options for the model
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',1, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'MiniBatchSize',4, ...
    'ExecutionEnvironment','gpu');
%% Train The CNN Network
net = trainNetwork(imdsTrain,layers,options);
%% Checking Network Performance
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);
%% Save the Network for future validation
save net;