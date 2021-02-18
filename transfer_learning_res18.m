 clc
 clear 


imds = imageDatastore('...\Turkey_PlantDataset\','IncludeSubfolders',true,'LabelSource','foldernames');
imds.ReadFcn = @(loc)imresize(imread(loc),[224,224]);
[trainImages,valImages] = splitEachLabel(imds,0.8,'randomized');


net = resnet18;

lgraph = layerGraph(net);


lgraph = removeLayers(lgraph, {'ClassificationLayer_predictions','prob','fc1000'});

numClasses = numel(categories(trainImages.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 20)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph,'pool5','fc');

options = trainingOptions('sgdm', ...
    'MiniBatchSize',8, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
        'Verbose',false, ...
    'Plots','training-progress');


net = trainNetwork(trainImages,lgraph,options);

predictedLabels = classify(net,valImages);
accuracy = mean(predictedLabels == valImages.Labels)