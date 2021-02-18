 clc
 clear 


imds = imageDatastore('C:\Users\DELL\Desktop\bitki hastalık\Turkey_PlantDataset\','IncludeSubfolders',true,'LabelSource','foldernames');
imds.ReadFcn = @(loc)imresize(imread(loc),[224,224]);
[trainImages,valImages] = splitEachLabel(imds,0.9,'randomized');

net = resnet50;

lgraph = layerGraph(net);


lgraph = removeLayers(lgraph, {'ClassificationLayer_fc1000','fc1000_softmax','fc1000'});

numClasses = numel(categories(trainImages.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 20)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph,'avg_pool','fc');

options = trainingOptions('sgdm', ...
    'MiniBatchSize',20, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',1e-4, ...
        'Verbose',false, ...
    'Plots','training-progress');


net = trainNetwork(trainImages,lgraph,options);

predictedLabels = classify(net,valImages);
accuracy = mean(predictedLabels == valImages.Labels)
