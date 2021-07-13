 clc
 clear 


imds = imageDatastore('...\Turkey_PlantDataset\','IncludeSubfolders',true,'LabelSource','foldernames');
imds.ReadFcn = @(loc)imresize(imread(loc),[224,224]);
[trainImages,valImages] = splitEachLabel(imds,0.8,'randomized');

net = densenet201;

lgraph = layerGraph(net);


lgraph = removeLayers(lgraph, {'ClassificationLayer_fc1000','fc1000_softmax','fc1000'});

numClasses = numel(categories(trainImages.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph,'avg_pool','fc');

options = trainingOptions('sgdm',...
    'MiniBatchSize',20,...
    'MaxEpochs',10,...
    'InitialLearnRate',1e-3,...
     'Verbose',false ,...
    'Plots','training-progress');


net = trainNetwork(trainImages,lgraph,options);

predictedLabels = classify(net,valImages);
accuracy = mean(predictedLabels == valImages.Labels)
