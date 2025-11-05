
%% Readme: Put this script in the same folder with the dataset_2025 folder

%% Set data folder and image labels
dataFolder = "dataset_2025"; %Adjust path if dataset is not in the same folder with this script
imds = imageDatastore(dataFolder, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");


    
classNames = categories(imds.Labels);
labelCount = countEachLabel(imds)
%% Check image size, to confirm size 128*128

img = readimage(imds,1);
size(img)
%% Split data to 75% (762) training and 25% validation
numTrainFiles = 762;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,"randomize");

% Data augmentation for training images
augmenter = imageDataAugmenter( ...
    RandRotation=[-30 30], ...
    RandXTranslation=[-20 20], RandYTranslation=[-20 20], ...
    RandXReflection=true);
 
augimdsTrain = augmentedImageDatastore([128 128 1], imdsTrain, ...
    DataAugmentation=augmenter);
%% Invert the color

pInvert = 0.5;  % 50% chance to invert
augimdsTrainInv = imageDatastore(augimdsTrain.Files, ...
    IncludeSubfolders=true, ...
    Labels=imdsTrain.Labels, ...
    ReadFcn=@(fname) maybeInvert(imread(fname), pInvert));
    %% Helper function to randomly invert training image black/white
function I = maybeInvert(I,p)
    if rand < p
        I = imcomplement(I);   % invert black/white
    end
end
%% Can preview what are the augmented images used for training

k = 25;
idx = randperm(numel(augimdsTrainInv.Files), k);

figure(904); clf
t = tiledlayout(5,5, "TileSpacing","compact", "Padding","compact");
title(t, "Random samples from imdsTrainInv (50% training images are inverted B/W)");

for i = 1:k
    I = readimage(augimdsTrainInv, idx(i));   % ReadFcn may invert this image
    nexttile;
    imshow(I, []);
    if ismatrix(I), colormap(gca, gray); end

    % Optional: show label (if available)
    if ~isempty(augimdsTrainInv.Labels)
        lbl = string(augimdsTrainInv.Labels(idx(i)));
        title(lbl, "FontSize", 8);
    end
end
drawnow
%% CNN architecture , adjustable (add in more layers or change filter number)

layers = [
    imageInputLayer([128 128 1])

    convolution2dLayer(3,16,Padding="same") 
    convolution2dLayer(3,16,Padding="same") 
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,Stride=2)

    convolution2dLayer(3,32,Padding="same")
    convolution2dLayer(3,32,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,Stride=2)

    convolution2dLayer(3,64,Padding="same")
    convolution2dLayer(3,64,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,Stride=2)         

    dropoutLayer(0.5)                
    fullyConnectedLayer(7)
    softmaxLayer
];
%% 

%% Training parameters, include some hyperparameters


options = trainingOptions("sgdm", ...
    InitialLearnRate=1e-3, ...
    LearnRateSchedule="piecewise", ...   % enable schedule
    LearnRateDropFactor=0.9, ...         % multiply LR by 0.9 at each drop
    LearnRateDropPeriod=2, ...           % drop every 2 epochs 
    MiniBatchSize=32, ...
    MaxEpochs=10, ...
    Shuffle="every-epoch", ...
    ValidationData=imdsValidation, ...
    ValidationFrequency=30, ...                 
    Metrics="accuracy", ...
    Plots="training-progress", ...
    Verbose=false);
% % %Above setting obtained 99.27% accuracy


%% Run this to get the CNN training, include training progress plot

net = trainnet(augimdsTrainInv,layers,"crossentropy",options);


%% 
% Evaluate the network on the validation set , use defined function below
visualize_result(imdsValidation,net,classNames)



%% Visualize function for validation set
function visualize_result(imageset, net, classNames)

scores = minibatchpredict(net, imageset);
YValidation = scores2label(scores, classNames);

TValidation = imageset.Labels;
accuracy = mean(YValidation == TValidation);

% Visualize random 25 validation images from the validation set
k = min(25, numel(imageset.Files));
if k > 0
    ix = randperm(numel(imageset.Files), k);
    figure(Name="Validation Visualization", NumberTitle="off"); 
    t = tiledlayout(5,5, "TileSpacing","compact", "Padding","compact");
    title(t, "Minibatch Prediction - 25 Validation Samples. (Random)", "FontWeight", "bold");
    subtitle(t, sprintf("Network Validation accuracy: %.2f%%", accuracy * 100));
    
    for i = 1:k
        j = ix(i);
        I = readimage(imageset, j);
        nexttile; imshow(I, []); 
        
        % Set color based on prediction match
        if YValidation(j) == imageset.Labels(j)
            color = "green"; % Green for correct prediction
        else
            color = "red";   % Red for incorrect prediction
        end
        
        title(sprintf("Predict:%s , Ground truth:%s", string(YValidation(j)), string(imageset.Labels(j))), "color", color, "FontSize", 9);
        
    end
end

% Visualize wrong predictions
wrongPredictions = find(YValidation ~= TValidation);
if ~isempty(wrongPredictions)
    figure(Name="Wrong Predictions Visualization", NumberTitle="off");
    t = tiledlayout(5,5, "TileSpacing","compact", "Padding","compact");
    title(t, "All Wrong Predictions - Validation Samples (Fixed)", "FontWeight", "bold");
    subtitle(t,"(Maxshow=25)");
    for i = 1:min(25, numel(wrongPredictions))
        j = wrongPredictions(i);
        I = readimage(imageset, j);
        nexttile; imshow(I, []); 
        
       title(sprintf("Predict:%s , Ground truth:%s", string(YValidation(j)), string(imageset.Labels(j))), "color", "red", "FontSize", 9);
    end
end

end

%% Run this to test part1 images in grayscale
%% 
% Unlabeled test folder (≤10 images)
part1pic = imageDatastore("seg_gray", IncludeSubfolders=true); %Your folder should be grayscale image from part1
part1aug = augmentedImageDatastore([128 128 1], part1pic, ...
    "OutputSizeMode","resize");

% scores = minibatchpredict(net, part1aug, "MiniBatchSize", 10);
scores = minibatchpredict(net, part1aug);
[conf, idx] = max(scores, [], 2);
YPred = categorical(classNames(idx), classNames);

N = numel(part1pic.Files);

figure("Name","Part1 predictions","NumberTitle","off"); 
tiledlayout(5, 5, "TileSpacing","compact", "Padding","compact");
for j = 1:N
    I = readimage(part1pic, j);    % show original (not resized) for readability
    nexttile; imshow(I, []); if ismatrix(I), colormap(gca, gray); end
    title(sprintf("Predict:%s (%.2f)", string(YPred(j)), conf(j)), "FontSize", 9);
end

%% 
function Iu8 = toGray255(fname)
    I = imread(fname);

    % 1) to grayscale
    if size(I,3) == 3
        I = rgb2gray(I);
    end

    % 2) to double [0,1]
    Id = im2double(I);

    % 3) If essentially binary, soften edges to create gray levels
    %    (can’t recover detail, but anti-aliasing helps)
    if islogical(I) || numel(unique(I)) <= 3
        Id = imgaussfilt(Id, 0.8);   % tweak sigma if you want softer/harder edges
    end

    % 4) Stretch and convert to uint8 [0..255]
    Id = mat2gray(Id);               % ensure full 0..1 span
    Iu8 = im2uint8(Id);              % -> 0..255 uint8
end

%% Use this to convert part1 image to grayscale (0-255) instead of using binary images
inFolder  = "segmented_characters"; %Change to your part1 folder name
outFolder = "seg_gray"; 
if ~exist(outFolder,"dir") 
    mkdir(outFolder); 
end
imds = imageDatastore(inFolder, IncludeSubfolders=true);
for i = 1:numel(imds.Files)
    Iu8 = toGray255(imds.Files{i});
    [~, name, ext] = fileparts(imds.Files{i});
    imwrite(Iu8, fullfile(outFolder, name + "_gray" + ext));
end

%% 

% Save trained model into folder
output_folder = 'Save_models';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end 
Advance_Net=net;
outputFile = fullfile(output_folder, "Advance_Net.mat");
save(outputFile,"Advance_Net");