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

labelCount = countEachLabel(imds);

%% Split data to 75% (762) training and 25% validation
numTrainFiles = 762;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,"randomize");

%% Load trained SVM model
output_folder = "Save_models";
outputFile = fullfile(output_folder,"SVM_Model.mat");
load(outputFile,"SVM_Model");

W = SVM_Model.W;
b = SVM_Model.b;
classNames = SVM_Model.classNames;
targetSize = SVM_Model.targetSize;
featDim = SVM_Model.featDim;
C = SVM_Model.C;
numEpochs = SVM_Model.numEpochs;
learningRate = SVM_Model.learningRate;

%% Extract features for validation set
[XVal,YValIdx] = extractFeature(imdsValidation, targetSize);
numVal = size(XVal,1);

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
for j = 1:min(N,25)
    I = readimage(part1pic,j);
    nexttile; imshow(I,[]); 
    if ismatrix(I), colormap(gca, gray); end
    title(sprintf("Predict:%s (%.2f)", string(YPred(j)),conf(j)),"FontSize",9);
end

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
            
            title(sprintf("Predict:%s , GT:%s", string(predLabel), string(trueLabel)), "color", color, "FontSize", 8);
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
            
            title(sprintf("Predict:%s , GT:%s", string(predLabel), string(trueLabel)), "color", "red", "FontSize", 8);
        end
    end
end
