# Abstract
Plant diseases and pests cause significant losses in agriculture, with economic, ecological and social implications. Therefore, early detection of plant diseases and pests via automated methods are very important. Recent machine learning-based studies have become popular in the solution of agricultural problems such as plant diseases. In this work, we present two classification models based on deep feature extraction from pretrained convolutional neural networks. In the proposed models, we fine-tune and combine six state-of-the-art convolutional neural networks and evaluate them on the given problem both individually and as an ensemble. Finally, the performances of different combinations based on the proposed models are calculated using an SVM classifier. In order to verify the validity of the proposed model, we collected Turkey-PlantDataset, consisting of unconstrained photographs of 15 kinds of disease and pest images observed in Turkey. According to the obtained performance results, the accuracy scores are calculated as 97.56% using the Majority Voting Ensemble model and 96.83% using the Early Fusion Ensemble model. The results demonstrate that the proposed models reach or exceed state-of-the-art results for this problem. 


# PlantDiseaseNet
PlantDiseaseNet: Convolutional Neural Network Ensemble for  Plant Disease and Pest Detection

The Turkey-PlantDataset called as Turkey Plant Diseases & Pests Dataset was obtained from academics working in the field of plant protection at the Agricultural Faculty of Bingol and Inonu Universities in Turkey. This dataset consists of a total of 4,447 images in 15 classes. The images in the dataset were obtained from experimental field studies of the Faculty of Agriculture at Inonu and Bingol Universities. The three-channel (RGB) color images were obtained using a Nikon 7200d camera, with an image resolution of 4000 x 6000 pixels. In addition, the dataset contains natural images including different scenarios such as soil, trees, leaves and sky. 

Turkey-PlantDataset dowloand link: https://drive.google.com/open?id=1gQNRL0I-udvAQXJV6Ewvhc14CLIgYmeA

