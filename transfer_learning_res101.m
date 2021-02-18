 clc
 clear 


images = imageDatastore('C:\Users\TÜRKOĞLU\Desktop\New folder\kayısı foto\','IncludeSubfolders',true,'LabelSource','foldernames');
images.ReadFcn = @(loc)imresize(imread(loc),[224,224]);
[trainImages,valImages] = splitEachLabel(images,0.8,'randomized');


net = resnet101;

lgraph = layerGraph(net);


lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});

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
