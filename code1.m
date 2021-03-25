digitDatasetPath = fullfile('E:\My Studys\Semester-3 03_11_2020--21_02_2020\Machine Learning\Project\masterIT');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

labelCount = countEachLabel(imds)

img = readimage(imds,1);
size(img)

numTrainFiles = 100;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

layers = [
    imageInputLayer([1077 1360 3])
    
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

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'MiniBatchSize',4, ...
    'ExecutionEnvironment','gpu');

net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
save net;