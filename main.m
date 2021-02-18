clear;clc;

layer ='fc6';
layer1 ='loss3-classifier';
layer2 ='fc1000';

net = alexnet;
net1 = googlenet;
net2 = resnet18;
net3 =resnet50;
net4 =resnet101;
net5 =densenet201;

%%
imds = imageDatastore('...\Turkey_PlantDataset\',...
    'IncludeSubfolders',true,...
    'LabelSource','FolderNames');
uzunluk=numel(imds.Labels);

for i=1:uzunluk
  
    img=readimage(imds,i);
    aa=size(img);
    if length(aa)==2
        img=cat(3,img,img,img);
    end    
   img=imresize(img,[224 224]);
   img1=imresize(img,[227 227]);

   alex_Feats(:,i) = activations(net,img1,layer);
   google_Feats(:,i) = activations(net1,img,layer1);
   res18_Feats(:,i) = activations(net2,img,layer2);
   res50_Feats(:,i) = activations(net3,img,layer2);
   res101_Feats(:,i) = activations(net4,img,layer2);
   dense_Feats(:,i) = activations(net5,img,layer2);

end
labels=imds.Labels;

%% PlantDiseaseNet-EF model
H=double(labels);
feat=[alex_Feats;google_Feats;dense_Feats;res18_Feats;res101_Feats;res50_Feats]';
YPred_EF=prediction(feat,H);
accuracy = mean(YPred_EF == Y1) // Results of PlantDiseaseNet-EF model


%% PlantDiseaseNet-Majority Voting model
Y1=double(labels);
YPred1=prediction(google_Feats,Y1);
YPred2=prediction(res18_Feats,Y1);
YPred3=prediction(res101_Feats,Y1);
YPred4=prediction(res50_Feats,Y1);
YPred5=prediction(dense_Feats,Y1);

for i=1:length(YPred1)
      diz=[YPred1(i);YPred2(i);YPred3(i);YPred4(i);YPred5(i)];
        YPredson(i)=mode(diz);
end

accuracy = mean(YPredson' == Y1) // Results of PlantDiseaseNet-MV model
