clc
clear 
net = alexnet;
 
imds = imageDatastore('...\Turkey_PlantDataset\','IncludeSubfolders',true,'LabelSource','foldernames');
imds.ReadFcn = @(loc)imresize(imread(loc),[227,227]);
[imdsTrain,idmsTest] = splitEachLabel(imds,0.8,'randomized');


layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm',...
    'MiniBatchSize',20,...
    'MaxEpochs',10,...
    'InitialLearnRate',1e-3,...
     'Verbose',false ,...
    'Plots','training-progress');
netTransfer = trainNetwork(imdsTrain,layers,options);

YPred = classify(netTransfer,idmsTest);
accuracy = mean(YPred == idmsTest.Labels)
