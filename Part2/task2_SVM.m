%% Task 2: non-CNN-based method using soft-margin SVM
% Readme: Put this script in the same folder with the dataset_2025 folder

clc;
clear;
close all;

%% Set data folder and image labels
dataFolder = "dataset_2025";
imds = imageDatastore(dataFolder, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

classNames = categories(imds.Labels);
labelCount = countEachLabel(imds);

%% Check image size, to confirm size 128*128
img = readimage(imds,1);
size(img);

%% Split data to 75% (762) training and 25% validation
numTrainFiles = 762;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,"randomize");

%% Extract features for training and validation
targetSize = [64 64];   % feature resolution
[XTrain,YTrainIdx] = extractFeature(imdsTrain, targetSize);
[XVal,YValIdx] = extractFeature(imdsValidation, targetSize);

numTrain = size(XTrain,1);
numVal = size(XVal,1);
numClasses = numel(classNames);
featDim = size(XTrain,2);

%% Train soft-margin SVM
C = 2.0;    % soft-margin penalty
numEpochs = 80;
learningRate = 1e-4;

W = zeros(featDim, numClasses);
b = zeros(1, numClasses);

for k = 1:numClasses
    fprintf('Training SVM for class %s \n', string(classNames(k)));
    
    yBinary = -ones(numTrain,1);
    yBinary(YTrainIdx == k) = 1;    % +1 for class k, -1 for others
    
    [wk, bk] = train_SVM(XTrain, yBinary, C, numEpochs, learningRate);
    W(:,k) = wk;
    b(k) = bk;
end

%% Evaluate on validation set
scoresVal = XVal * W + repmat(b,numVal,1);
[~, predIdxVal] = max(scoresVal,[],2);

valAccuracy = mean(predIdxVal == YValIdx);
fprintf('Validation accuracy: %.2f%%\n', valAccuracy*100);

VisualizeResult(imdsValidation, predIdxVal, YValIdx, classNames);

%% Test part1 images
part1pic = imageDatastore("seg_gray", IncludeSubfolders=true);
[XPart1, ~] = extractFeature(part1pic,targetSize,false);

N = size(XPart1,1);
scoresPart1 = XPart1 * W + repmat(b,N,1);
[conf, idxPart1] = max(scoresPart1,[],2);
YPred = classNames(idxPart1);

figure("Name","Part1 SVM predictions","NumberTitle","off");
tiledlayout(5, 5, "TileSpacing","compact", "Padding","compact");
for j = 1:N
    I = readimage(part1pic,j);     % show original (not resized) for readability
    nexttile; imshow(I,[]); if ismatrix(I), colormap(gca, gray); end
    title(sprintf("Predict:%s (%.2f)", string(YPred(j)),conf(j)),"FontSize",9);
end

%% Use this to convert part1 images to grayscale (0-255) instead of using binary images
% (Run once before training/testing if you only have binary segmented images)
inFolder  = "segmented_characters";
outFolder = "seg_gray"; 
if ~exist(outFolder,"dir")
    mkdir(outFolder);
end

imdsPart1 = imageDatastore(inFolder,IncludeSubfolders=true);
for i = 1:numel(imdsPart1.Files)
    Iu8 = toGray255(imdsPart1.Files{i});
    [~, name, ext] = fileparts(imdsPart1.Files{i});
    imwrite(Iu8, fullfile(outFolder,name + "_gray" + ext));
end

%% Save trained SVM model into folder
output_folder = 'Save_models';
if ~exist(output_folder,'dir')
    mkdir(output_folder);
end
SVM_Model.W = W;
SVM_Model.b = b;
SVM_Model.classNames = classNames;
SVM_Model.targetSize = targetSize;
SVM_Model.featDim = featDim;
SVM_Model.C = C;
SVM_Model.numEpochs = numEpochs;
SVM_Model.learningRate = learningRate;

outputFile = fullfile(output_folder,"SVM_Model.mat");
save(outputFile,"SVM_Model");


%% Feature extraction for the dataset
function [X, YIdx] = extractFeature(imds, targetSize, useLabels)
    if nargin < 3
        useLabels = true;
    end
    
    numImages = numel(imds.Files);
    sampleImg = readimage(imds,1);
    sampleFeat = extractSingleFeature(sampleImg,targetSize);
    featDim = numel(sampleFeat);
    
    X = zeros(numImages,featDim);
    if useLabels
        YIdx = zeros(numImages,1);
    else
        YIdx = [];
    end
    
    for i = 1:numImages
        I = readimage(imds, i);
        X(i,:) = extractSingleFeature(I, targetSize);
        if useLabels
            YIdx(i) = double(imds.Labels(i));
        end
    end
end


%% Feature extraction for single image
function feat = extractSingleFeature(I, targetSize)
    if size(I,3) == 3
        I = 0.2989*I(:,:,1) + 0.5870*I(:,:,2) + 0.1140*I(:,:,3);
    end
    
    I = double(I);
    I = imresize(I, targetSize);
    I = I - min(I(:));
    if max(I(:)) > 0
        I = I ./ max(I(:));
    end
    feat = I(:)';   % row vector
end



%% train linear soft-margin SVM
function [w, b] = train_SVM(X, y, C, numEpochs, eta0)
    [N, D] = size(X);
    w = zeros(D,1);
    b = 0;
    
    for epoch = 1:numEpochs
        eta = eta0 / (1 + 0.1*(epoch-1));   % learning rate
        
        idx = randperm(N);
        Xshuf = X(idx,:);
        yshuf = y(idx);
        
        for i = 1:N
            xi = Xshuf(i,:)';
            yi = yshuf(i);
            
            margin = yi * (w' * xi + b);
    
            if margin < 1
                w = w - eta * (w - C * yi * xi);
                b = b + eta * C * yi;
            else
                w = w - eta * w;
            end
        end
    end
end


%% Visualize validation predictions
function VisualizeResult(imdsValidation, predIdxVal, YValIdx, classNames)
    N = numel(imdsValidation.Files);
    accuracy = mean(predIdxVal == YValIdx);
    
    % Visualize random 25 validation images from the validation set
    k = min(25,N);
    if k > 0
        ix = randperm(N,k);
        figure(Name="Validation Visualization (SVM)", NumberTitle="off");
        t = tiledlayout(5,5, "TileSpacing","compact", "Padding","compact");
        title(t, "SVM Prediction - 25 Validation Samples (Random)", "FontWeight", "bold");
        subtitle(t, sprintf("Validation accuracy: %.2f%%", accuracy * 100));
        
        for i = 1:k
            j = ix(i);
            I = readimage(imdsValidation,j);
            nexttile; imshow(I,[]); 
            if ismatrix(I), colormap(gca, gray); end
            
            predLabel = classNames(predIdxVal(j));
            trueLabel = classNames(YValIdx(j));
            
            if predIdxVal(j) == YValIdx(j)
                color = "green";    % Green for correct prediction
            else
                color = "red";      % Red for incorrect prediction
            end
            
            title(sprintf("Predict:%s , Ground truth:%s", string(predLabel), string(trueLabel)), "color", color, "FontSize", 8);
        end
    end
    
    % Visualize wrong predictions
    wrongIdx = find(predIdxVal ~= YValIdx);
    if ~isempty(wrongIdx)
        figure(Name="Wrong Predictions (SVM)", NumberTitle="off");
        t = tiledlayout(5,5, "TileSpacing","compact", "Padding","compact");
        title(t, "Wrong Predictions - Validation Samples (SVM, max 25)", "FontWeight", "bold");
        
        for i = 1:min(25, numel(wrongIdx))
            j = wrongIdx(i);
            I = readimage(imdsValidation, j);
            nexttile; imshow(I, []); 
            if ismatrix(I), colormap(gca, gray); end
            
            predLabel = classNames(predIdxVal(j));
            trueLabel = classNames(YValIdx(j));
            
            title(sprintf("Predict:%s , Ground truth:%s", string(predLabel), string(trueLabel)), "color", "red", "FontSize", 8);
        end
    end
end


%% Turn image into grayscale type
function Iu8 = toGray255(fname)
    I = imread(fname);
    
    if size(I,3) == 3
        I = 0.2989*I(:,:,1) + 0.5870*I(:,:,2) + 0.1140*I(:,:,3);
    end
    
    Id = im2double(I);
    
    if islogical(I) || numel(unique(I)) <= 3
        Id = imgaussfilt(Id, 0.8);
    end
    
    Id = mat2gray(Id);               % ensure full 0..1 span
    Iu8 = im2uint8(Id);              % -> 0..255 uint8
end
