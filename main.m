clear all;
close all;

% loading pretrainedNetwork
% pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/deeplabv3plusResnet18CamVid.mat';
% pretrainedFolder = fullfile(tempdir,'pretrainedNetwork');
% pretrainedNetwork = fullfile(pretrainedFolder,'deeplabv3plusResnet18CamVid.mat'); 
% if ~exist(pretrainedNetwork,'file')
%     mkdir(pretrainedFolder);
%     disp('Downloading pretrained network (58 MB)...');
%     websave(pretrainedNetwork,pretrainedURL);
% end

%loading data
dataFolder = 'D:\SungRung\mnist_SEG(Noise)\project\data\noiseMnist';
imds = imageDatastore(dataFolder,'IncludeSubfolders',true,'LabelSource','foldernames', 'ReadFcn', @to3D);
[imdsTrain, imdsValid] = splitEachLabel(imds, 0.7, 'randomized');
numTrainImages = numel(imdsTrain.Labels);
 
% %loading pixel-label images

classes = ["background" "edge"];
labelIDs = camvidPixelLabelIDs();
labelDir = "D:\SungRung\mnist_SEG(Noise)\project\data\resizedMnist"
pxds =pixelLabelDatastore(labelDir, classes, labelIDs, 'ReadFcn', @to2D);

cmap = camvidColorMap;

tbl = countEachLabel(pxds);

[imdsTrain, imdsVal, imdsTest, pxdsTrain,pxdsVal, pxdsTest] = partitionCamVidData(imds, pxds);

imageSize = [720 960 3];
numClasses = numel(classes);
% lgraph = deeplabv3plusLayers(imageSize, numClasses, 'resnet18');
 
 %balance class weights
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./imageFreq;

pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);

lgraph = unetLayers(imageSize, numClasses);

ds = combine(imds,pxds);

% Define training options. 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-2, ...
    'L2Regularization',0.005, ...
    'MaxEpochs',1, ...  
    'MiniBatchSize',1, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'Plots','training-progress');

% data augmentation 
augmenter = imageDataAugmenter('RandXReflection', true, 'RandXTranslation', [-100 100],'RandYTranslation', [-100 100]);


% pximds = pixelLabelImageDatastore(imdsTrain, pxdsTrain, 'DataAugmentation', augmenter);

numel(imdsTrain.Labels)
numel(imdsVal.Labels)
doTraining = true;
name = "Unet";
second = ".mat";
for i = 1: 2
    if doTraining    
        disp("Training.....")
        network = name + i + second;
        [net, info] = trainNetwork(ds,lgraph,options);

        save(network, 'net', 'info', 'options');
        disp("Trained")
    else
        data = load("Unet.mat")
        net = data.net;
    end
end


%functions//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
function labelIDs = camvidPixelLabelIDs()

labelIDs = { 0 %black
             1 %white
    };
end

function [imdsTrain, imdsVal, imdsTest, pxdsTrain,pxdsVal, pxdsTest] = partitionCamVidData(imds,pxds)
% Partition CamVid data by randomly selecting 60% of the data for training. The
% rest is used for testing.
%     pause; 
% Set initial random state for example reproducibility.
rng(0);

numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Use 60% of the images for training.
numTrain = round(0.015 * numFiles);
trainingIdx = shuffledIndices(1:numTrain);

% Use 20% of the images for validation
numVal = round(0.1 * numFiles);
valIdx = shuffledIndices(numTrain+1:numTrain+numVal);

% Use the rest for testing.
testIdx = shuffledIndices(numTrain+numVal+1:end);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
valImages = imds.Files(valIdx);
testImages = imds.Files(testIdx);

imdsTrain = imageDatastore(trainingImages,'ReadFcn', @to3D);
imdsVal = imageDatastore(valImages,'ReadFcn', @to3D);
imdsTest = imageDatastore(testImages,'ReadFcn', @to3D);

% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = camvidPixelLabelIDs();

% Create pixel label datastores for training and test.

trainingLabels = pxds.Files(trainingIdx);


valLabels = pxds.Files(valIdx);

testLabels = pxds.Files(valIdx);
% 111
% pause;
pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs, 'ReadFcn', @to2D);
pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs, 'ReadFcn', @to2D);
pxdsTest =pixelLabelDatastore(valLabels, classes, labelIDs, 'ReadFcn', @to2D);


end

function img = to3D(file)
     img = imread(file);
     if (size(img, 3) == 1)
     img= repmat(img, [1 1 3]);
     end

end

function img = to2D(file)
     img = imread(file);
     if (size(img, 3) == 3)
    img = rgb2gray(img);
     end

end

function cmap = camvidColorMap()
    cmap = [10 10 10
            240 240 240
           ];

    cmap = cmap ./ 255;
end

function pixelLabelColorbar(cmap, classNames)
% Add a colorbar to the current axis. The colorbar is formatted
% to display the class names with the color.

colormap(gca,cmap)

% Add colorbar to current figure.
c = colorbar('peer', gca);

% Use class names for tick marks.
c.TickLabels = classNames;
numClasses = size(cmap,1);

% Center tick labels.
c.Ticks = 1/(numClasses*2):1/numClasses:1;

% Remove tick mark.
c.TickLength = 0;
end







