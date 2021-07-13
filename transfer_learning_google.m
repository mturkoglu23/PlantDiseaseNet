clc
 clear 


imds = imageDatastore('...\Turkey_PlantDataset\','IncludeSubfolders',true,'LabelSource','foldernames');
imds.ReadFcn = @(loc)imresize(imread(loc),[224,224]);
[trainImages,valImages] = splitEachLabel(imds,0.9,'randomized');


net = googlenet;

inputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net);


lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});

numClasses = numel(categories(trainImages.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');


options = trainingOptions('sgdm',...
    'MiniBatchSize',20,...
    'MaxEpochs',10,...
    'InitialLearnRate',1e-3,...
     'Verbose',false ,...
    'Plots','training-progress');

net = trainNetwork(trainImages,lgraph,options);


YPred = classify(net,valImages);
accuracy = mean(YPred == valImages.Labels)
