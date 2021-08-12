imageSize = [720, 960 1];
numClasses = 5;
encoderDepth = 3;
lgraph = unetLayers(imageSize,numClasses);

plot(lgraph);

%loading data
dataFolder = 'D:\SungRung\mnist_SEG(Noise)\project\data\noiseMnist';
imds = imageDatastore(dataFolder,'IncludeSubfolders',true,'LabelSource','foldernames', 'ReadFcn', @to3D);


classes = ["background" "edge"];
labelIDs = { 0 %black
             1 %white
    };

labelDir = "D:\SungRung\mnist_SEG(Noise)\project\data\resizedMnist";
pxds =pixelLabelDatastore(labelDir, classes, labelIDs, 'ReadFcn', @to2D);

imageSize = [720, 960 3];
numClasses = 2;
lgraph = unetLayers(imageSize, numClasses);


options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...
    'MaxEpochs',1, ...  
    'MiniBatchSize',1, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress');


% doTraining = true;
% name = "Unet";
% second = ".mat";
% for i = 1: 2
%     if doTraining    
%         disp("Training.....")
%         network = name + i + second;
%          [imdsTrain, imdsVal, imdsTest, pxdsTrain,pxdsVal, pxdsTest] = partitionCamVidData(imds, pxds);
%           ds = combine(imdsTrain,pxdsTrain);
%         [net, info] = trainNetwork(ds,lgraph,options);
% 
%         save(network, 'net', 'info', 'options');
%         disp("Trained")
%     else
%         data = load("Unet.mat")
%         net = data.net;
%     end
% end

%testing image

image1 = readimage(imds,1);

model1 = load("Unet1.mat");
model2 = load("Unet2.mat");
unetModel1 = model1.net; 
unetModel2 = model2.net; 
segImg1 = semanticseg(image1, unetModel1); 
segImg2 = semanticseg(image1, unetModel2); 
output1 = cast(segImg1, 'double')-1;
output2 = cast(segImg1, 'double')-1;
% 
% subplot(1,3,1);
% imshow(image1);
% 
subplot(1,2,1);
imshow(output1);

subplot(1,2,2);
imshow(output2);


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


function [imdsTrain, imdsVal, imdsTest, pxdsTrain,pxdsVal, pxdsTest] = partitionCamVidData(imds,pxds)

    rng(0);

    numFiles = numel(imds.Files);
    shuffledIndices = randperm(numFiles);

    % Use 2% of the images for training.
    numTrain = round(0.02 * numFiles);
    trainingIdx = shuffledIndices(1:numTrain);

    % Use 1% of the images for validation
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

    pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs, 'ReadFcn', @to2D);
    pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs, 'ReadFcn', @to2D);
    pxdsTest =pixelLabelDatastore(valLabels, classes, labelIDs, 'ReadFcn', @to2D);

end

function labelIDs = camvidPixelLabelIDs()

labelIDs = { 0 %black
             1 %white
    };
end


