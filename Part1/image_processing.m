%% ME5411 Project - Character extraction (Steps 1–6) with Tabbed Visualization
clear; clc; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1. Read image and create contrast-enhanced variants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read BMP and ensure grayscale uint8
I = imread("charact2.bmp");

if size(I,3) == 3
    I = rgb2gray(I);
end
I = im2uint8(I);

% Contrast variants
sl = stretchlim(I, [0.01 0.99]);
I_imadjust    = imadjust(I, sl, []);
I_histeq      = histeq(I);
I_adapthisteq = adapthisteq(I, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
col_titles    = {'Original', 'imadjust', 'histeq', 'adapthisteq'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 2. Smoothing (box filters, bilateral, and combined variants)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filter_sizes     = [3, 5, 7];
num_rows         = numel(filter_sizes) + 3;  % rows: No filter, 3 box sizes, Bilateral, 5x5+Bilateral
filtered_images  = cell(num_rows, 4);

% Row 1: No filter (original variants)
filtered_images(1,:) = {I, I_imadjust, I_histeq, I_adapthisteq};

% Rows 2–4: Box (averaging) filter variants
for i = 1:numel(filter_sizes)
    sz = filter_sizes(i);
    filtered_images{i+1,1} = imboxfilt(I, sz);
    filtered_images{i+1,2} = imboxfilt(I_imadjust, sz);
    filtered_images{i+1,3} = imboxfilt(I_histeq, sz);
    filtered_images{i+1,4} = imboxfilt(I_adapthisteq, sz);
end

% Row 5: Bilateral on originals
for c = 1:4
    filtered_images{numel(filter_sizes)+2, c} = im2uint8(imbilatfilt(im2double(filtered_images{1,c})));
end

% Row 6: 5x5 box then bilateral
for c = 1:4
    filtered_images{numel(filter_sizes)+3, c} = im2uint8(imbilatfilt(im2double(filtered_images{3,c}))); % row 3 corresponds to 5x5
end

% Define row titles for Step 2 visualization
row_titles = {'No Filter','3x3','5x5','7x7','Bilateral','5x5+Bilateral'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 3. Crop region containing "HD44780A00"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rect = [51.51, 207.51, 896.98, 129.98];
rows_to_crop  = [3, 5];  
num_cols      = size(filtered_images,2);
cropped_images = cell(numel(rows_to_crop), num_cols);

for rr = 1:numel(rows_to_crop)
    for c = 1:num_cols
        cropped_images{rr,c} = imcrop(filtered_images{rows_to_crop(rr), c}, rect);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 4. Convert cropped images to binary (invert then per-image thresholds)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
R = size(cropped_images,1);
C = size(cropped_images,2);

T_mode = 'uint8';     
T_list = [115, 145, 203, 145, 120, 179, 229, 126];

BW_thresh = cell(R, C);
T_grid    = zeros(R, C);

k = 1;
for rr = 1:R
    for c = 1:C
        Iorig = cropped_images{rr,c};
        if ~isa(Iorig,'uint8'), Iorig = im2uint8(Iorig); end

        T = T_list(k);
        if strcmpi(T_mode,'norm')
            T_val = uint8(255 * min(max(T,0),1));
        else
            T_val = uint8(min(max(T,0),255));
        end

        % Threshold directly on original image (white fg on black bg)
        BW = Iorig >= T_val;        
        BW_thresh{rr,c} = BW;      
        T_grid(rr,c)    = double(T_val);
        k = k + 1;
    end
end

% Define row titles for Step 4 visualization
row_titles2 = {'5x5','Bilateral'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 5. Character cleaning, bolding, and outline generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
assert(exist('BW_thresh','var')==1, 'Run Step 4 first.');
targets        = [2, 3];   
minAreaBlack   = 200;      
conn           = 4;

BW_thresh_before = BW_thresh;

for c = targets
    rr = 1;
    BW = BW_thresh{rr,c};  
    % Remove small white noise (foreground)
    BW = bwareaopen(BW, minAreaBlack, conn); 
    BW_thresh{rr,c} = BW;
end

rr = 1; cc = 3;
minAreaBlack = 50; conn = 8; neckSize = 3; 

BW = BW_thresh{rr,cc};  

seBreak = strel('disk', neckSize);
E = imerode(BW, seBreak);  % Erode white foreground to break necks
BW_tildeE = E;             

% Remove small white islands (noise)
black_islands = BW_tildeE;  
black_large   = bwareaopen(black_islands, minAreaBlack, conn);
black_small   = black_islands & ~black_large;

BW_tildeE_clean = BW_tildeE & ~black_small;  % Remove small islands
BW_clean_simple = BW_tildeE_clean;

if ~exist('BW_neckcut','var'), BW_neckcut = BW_thresh; end
BW_neckcut{rr,cc} = BW_clean_simple;

BW_src = BW_clean_simple;      

% Bolding variants
BW_boldA = cell(R,C); BW_boldB = cell(R,C); BW_boldC = cell(R,C); BW_boldD = cell(R,C);

rA = 1;
BW_boldA{rr,cc} = imdilate(BW_src, strel('disk', rA));

cleanAreaB = 60; rB = 2; itB = 1; rShrink = 1;
F_B = BW_src;
if cleanAreaB > 0
    F_B = bwareaopen(F_B, cleanAreaB);
end
for i = 1:itB
    F_B = imdilate(F_B, strel('disk', rB));
end
F_B = imerode(F_B, strel('disk', rShrink));
BW_boldB{rr,cc} = F_B;

rGrow = 2; rShrink = 1;
BW_boldC{rr,cc} = imerode(imdilate(BW_src, strel('disk', rGrow)), strel('disk', rShrink));
BW_C = BW_boldC{rr,cc};

rSk = 1;
Sk  = bwmorph(BW_src, 'skel', Inf);
BW_boldD{rr,cc} = imdilate(Sk, strel('disk', rSk));

base_inv = ~BW_clean_simple;          
BW_B = BW_boldB{rr,cc};               

perimB = bwperim(BW_B);
perimB_thick = imdilate(perimB, strel('disk', 1));

overlay_on_B_thin  = imoverlay(BW_B, perimB, [1 0 0]);          
overlay_on_B_thick = imoverlay(BW_B, perimB_thick, [1 0 0]);

perimC = bwperim(BW_boldC{rr,cc});                              
overlay_from_C     = imoverlay(BW_clean_simple, perimC, [1 0 0]);       
overlay_from_C_inv = imcomplement(overlay_from_C);               
perimC_thick = imdilate(perimC, strel('disk', 1));
overlay_from_C_inv_outlined_thin  = imoverlay(rgb2gray(overlay_from_C_inv), perimC,       [1 0 0]);
overlay_from_C_inv_outlined_thick = imoverlay(rgb2gray(overlay_from_C_inv), perimC_thick, [1 0 0]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 6. Character segmentation and bounding box extraction on chosen mask
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% I_bs = ~BW_clean_simple; 
I_bs = BW_B;

[L1, num] = bwlabel(I_bs);
props = regionprops(L1, 'BoundingBox');
fprintf('Initial detection: %d characters\n', num);

RGB_label = label2rgb(L1, 'jet', 'k', 'shuffle');
[L2, num2, props2] = split_stuck_characters(I_bs, props);
RGB_label2 = label2rgb(L2, 'jet', 'k', 'shuffle');

% Create output folder if it doesn't exist, and save each character crop
output_folder = 'segmented_characters';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

for k = 1:num2
    bb = round(props2(k).BoundingBox);
    x1 = max(1, bb(1));
    y1 = max(1, bb(2));
    x2 = min(size(I_bs, 2), x1 + bb(3) - 1);
    y2 = min(size(I_bs, 1), y1 + bb(4) - 1);
    
    % Extract character (binary)
    char_img = I_bs(y1:y2, x1:x2);
    char_bin = logical(char_img);

    % If border pixels are mostly foreground, invert so background=0, foreground=1
    border_pixels = [char_bin(1,:), char_bin(end,:), char_bin(:,1)', char_bin(:,end)'];
    if mean(border_pixels) > 0.5
        char_bin = ~char_bin;
    end

    % Optionally resize to fit into 128x128 without changing aspect ratio
    target_size = 128;
    h = size(char_bin,1); w = size(char_bin,2);
    scale = min(1, target_size / max(h,w));
    if scale < 1
        char_bin = imresize(char_bin, scale, 'nearest');
        h = size(char_bin,1); w = size(char_bin,2);
    end

    % Create white canvas and center the character (character will be black on white)
    canvas = true(target_size, target_size);
    r0 = floor((target_size - h)/2) + 1;
    c0 = floor((target_size - w)/2) + 1;
    r1 = r0 + h - 1;
    c1 = c0 + w - 1;

    region = canvas(r0:r1, c0:c1);
    region(char_bin) = false;
    canvas(r0:r1, c0:c1) = region;

    filename = fullfile(output_folder, sprintf('char_%02d.png', k));
    imwrite(canvas, filename);
end

fprintf('Saved %d character images to folder "%s"\n', num2, output_folder);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create one figure with tabs for all steps visualization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig = figure('Name', 'ME5411 Project Visualization', 'Position', [100 100 1400 900]);
tg = uitabgroup(fig);

% Step 1 Tab
tab1 = uitab(tg, 'Title', 'Step 1 - Contrast');
ax1 = axes('Parent', tab1);
ax1.Position = [0.1 0.5 0.8 0.4];
montage({I, I_imadjust, I_histeq, I_adapthisteq}, 'Size', [1 4], 'Parent', ax1);
title(ax1, 'Original | imadjust | histeq | adapthisteq');

% Step 2 Tab
tab2 = uitab(tg, 'Title', 'Step 2 - Smoothing');
t2 = tiledlayout(tab2, num_rows, numel(col_titles), 'TileSpacing', 'compact', 'Padding', 'compact');
for r = 1:num_rows
    for c = 1:numel(col_titles)
        ax = nexttile(t2);
        imshow(filtered_images{r,c}, 'Parent', ax);
        if r == 1, title(ax, col_titles{c}); end
        if c == 1, ylabel(ax, row_titles{r}, 'FontWeight', 'bold'); end
    end
end

% Step 3 Tab
row_titles3 = {'5x5 Box Filter', 'Bilateral Filter'};
tab3 = uitab(tg, 'Title', 'Step 3 - Cropped');
t3 = tiledlayout(tab3, numel(rows_to_crop), num_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
for rr = 1:numel(rows_to_crop)
    for c = 1:num_cols
        ax = nexttile(t3);
        imshow(cropped_images{rr,c}, 'Parent', ax);
        if rr == 1
            title(ax, col_titles{c});
        end
        if c == 1
            ylabel(ax, row_titles3{rr}, 'FontWeight', 'bold');
        end
    end
end

% Step 4 Tab
tab4 = uitab(tg, 'Title', 'Step 4 - Thresholding');
t4 = tiledlayout(tab4, R, C*3, 'TileSpacing', 'compact', 'Padding', 'compact');
for rr = 1:R
    for c = 1:C
        % Show original cropped image
        ax1 = nexttile(t4);
        imshow(cropped_images{rr,c}, 'Parent', ax1);
        if rr == 1, title(ax1, sprintf('%s (orig)', col_titles{c})); end
        if c == 1, ylabel(ax1, row_titles2{rr}, 'FontWeight', 'bold'); end

        % Show binary thresholded image
        ax2 = nexttile(t4);
        imshow(BW_thresh{rr,c}, 'Parent', ax2);
        title(ax2, sprintf('Binary @ T=%d', T_grid(rr,c)));

        % Show histogram with threshold line
        ax3 = nexttile(t4);
        Iorig = cropped_images{rr,c};
        if ~isa(Iorig,'uint8'), Iorig = im2uint8(Iorig); end
        [cnt,bins] = imhist(Iorig);
        bar(bins,cnt,'FaceColor',[0.8 0.2 0.2],'EdgeColor','none', 'Parent', ax3);
        yl = ylim(ax3);
        hold(ax3, 'on');
        plot(ax3, [T_grid(rr,c) T_grid(rr,c)], yl, 'k--', 'LineWidth', 1.5);
        hold(ax3, 'off');
        xlim(ax3, [0 255]);
        title(ax3, 'Hist + T');
        xlabel(ax3, 'Intensity');
        ylabel(ax3, 'Count');
    end
end

% Step 5 Tab
tab5 = uitab(tg, 'Title', 'Step 5 - Cleaning & Bolding');
num_rows = 4;
num_cols = 8;  % 1 extra column for row legend
t5 = tiledlayout(tab5, num_rows, num_cols, 'TileSpacing', 'compact', 'Padding', 'compact');

% Row legends (leftmost column)
row_legends = {
    'Orig & Before/After', ...
    'Cleaning & Bolding Part 1', ...
    'Bolding & Outlines Part 2', ...
    'Overlays & Outlines'
};

% Add row legends
for r = 1:num_rows
    add_row_legend(t5, r, row_legends{r}, num_cols);
end

% Row 1 (start from tile 2 to 8)
imshow(cropped_images{1,2}, 'Parent', nexttile(t5)); title('Orig imadjust');
imshow(BW_thresh_before{1,2}, 'Parent', nexttile(t5)); title('Before');
imshow(BW_thresh{1,2}, 'Parent', nexttile(t5)); title('After');

imshow(cropped_images{1,3}, 'Parent', nexttile(t5)); title('Orig histeq');
imshow(BW_thresh_before{1,3}, 'Parent', nexttile(t5)); title('Before');
imshow(BW_thresh{1,3}, 'Parent', nexttile(t5)); title('After');

imshow(BW, 'Parent', nexttile(t5)); title('Input BW (r1,c3)');

% Row 2
imshow(E, 'Parent', nexttile(t5)); title(sprintf('E (break disk=%d)', neckSize));
imshow(black_small, 'Parent', nexttile(t5)); title(sprintf('Small WHITE dots (<= %d px)', minAreaBlack));
imshow(BW_clean_simple, 'Parent', nexttile(t5)); title('BW after removal');
imshow(BW_clean_simple, 'Parent', nexttile(t5)); title('Cleaned (r1,c3)');
imshow(BW_boldA{1,3}, 'Parent', nexttile(t5)); title(sprintf('Bold A: d%d', rA));
imshow(BW_boldB{1,3}, 'Parent', nexttile(t5)); title(sprintf('Bold B: r=%d,it=%d,clean=%d', rB, itB, cleanAreaB));
imshow(BW_boldC{1,3}, 'Parent', nexttile(t5)); title('Bold C');

% Row 3
imshow(BW_boldD{1,3}, 'Parent', nexttile(t5)); title('Bold D');
imshow(imoverlay(BW_clean_simple, bwperim(BW_boldC{1,3}), [1 0 0]), 'Parent', nexttile(t5)); title('C vs Clean (perim)');
imshow(imoverlay(BW_clean_simple, bwperim(BW_boldB{1,3}), [1 0 0]), 'Parent', nexttile(t5)); title('B vs Clean (perim)');
imshow(BW_boldB{1,3}, 'Parent', nexttile(t5)); title('Bold B');
imshow(perimB, 'Parent', nexttile(t5)); title('Bold B perimeter');
imshow(overlay_on_B_thin, 'Parent', nexttile(t5)); title('Bold B + outline (thin)');
imshow(overlay_on_B_thick, 'Parent', nexttile(t5)); title('Bold B + outline (thick)');

% Row 4
imshow(overlay_from_C, 'Parent', nexttile(t5)); title('Overlay from C (base + perimC)');
imshow(overlay_from_C_inv, 'Parent', nexttile(t5)); title('Overlay from C (inverted)');
imshow(overlay_from_C_inv_outlined_thin, 'Parent', nexttile(t5)); title('Inverted + outline (thin, perimC)');
imshow(overlay_from_C_inv_outlined_thick, 'Parent', nexttile(t5)); title('Inverted + outline (thick, perimC)');

% Fill remaining tiles with empty axes to keep layout consistent
total_tiles = num_rows * num_cols;
used_tiles = 4*7 + 4;  % 4 rows * 7 images + 4 legends = 32, but we have only 32 tiles total
% Actually, we have 4 rows * 8 cols = 32 tiles total
% We used 4 legends + 28 images = 32 tiles total, so no empty tiles needed
% But if you want to be safe, you can fill any remaining tiles:

remaining_tiles = total_tiles - used_tiles;
for i = 1:remaining_tiles
    ax = nexttile(t5);
    axis(ax, 'off');
end

% Step 6 Tab
tab6 = uitab(tg, 'Title', 'Step 6 - Segmentation');
t6 = tiledlayout(tab6, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% Initial Labeling with bounding boxes
ax = nexttile(t6);
imshow(RGB_label, 'Parent', ax);
title(ax, sprintf('Initial Labeling: %d characters detected', num));
hold(ax, 'on');
for k = 1:num
    rectangle('Position', props(k).BoundingBox, 'EdgeColor', 'g', 'LineWidth', 1, 'Parent', ax);
    text(props(k).BoundingBox(1), props(k).BoundingBox(2)-5, ...
         num2str(k), 'Color', 'red', 'FontSize', 15, 'FontWeight', 'bold', 'Parent', ax);
end
hold(ax, 'off');

% After Splitting with bounding boxes
ax = nexttile(t6);
imshow(RGB_label2, 'Parent', ax);
title(ax, sprintf('After Splitting: %d characters detected', num2));
hold(ax, 'on');
for k = 1:num2
    rectangle('Position', props2(k).BoundingBox, 'EdgeColor', 'g', 'LineWidth', 1, 'Parent', ax);
    text(props2(k).BoundingBox(1), props2(k).BoundingBox(2)-5, ...
         num2str(k), 'Color', 'red', 'FontSize', 15, 'FontWeight', 'bold', 'Parent', ax);
end
hold(ax, 'off');

% Inverted Binary with bounding boxes (existing)
ax = nexttile(t6);
imshow(~I_bs, 'Parent', ax);
title(ax, 'Inverted Binary with Bounding Boxes and Labels');
hold(ax, 'on');
for k = 1:num2
    rectangle('Position', props2(k).BoundingBox, 'EdgeColor', rand(1,3), 'LineWidth', 1, 'Parent', ax);
    text(props2(k).BoundingBox(1), props2(k).BoundingBox(2)-5, ...
         num2str(k), 'Color', 'red', 'FontSize', 15, 'FontWeight', 'bold', 'Parent', ax);
end
hold(ax, 'off');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions (unchanged)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = B_final_or(BW)
    out = uint8(BW) * 255;
end

function r = getradius(seObj)
    try
        n = seObj.getnhood;
        [yy,xx] = ndgrid(1:size(n,1),1:size(n,2));
        cy = (size(n,1)+1)/2; cx = (size(n,2)+1)/2;
        rr2 = ((yy-cy).^2 + (xx-cx).^2) .* double(n);
        r = round(sqrt(max(rr2(:))));
    catch
        r = 0;
    end
end

function [BW_new, num_new, props_new] = split_stuck_characters(BW, props)
    BW_new = false(size(BW));
    widths = arrayfun(@(x)x.BoundingBox(3), props);
    avg_width = median(widths);
    threshold = 1.3 * avg_width;

    new_boxes = [];

    for k = 1:length(props)
        bb = round(props(k).BoundingBox);
        x1 = max(1, bb(1)); 
        y1 = max(1, bb(2));
        x2 = min(size(BW,2), x1 + bb(3) - 1);
        y2 = min(size(BW,1), y1 + bb(4) - 1);
        sub = BW(y1:y2, x1:x2);

        if bb(3) > threshold
            mid_start = round(bb(3)*0.4);
            mid_end = round(bb(3)*0.6);
            mid_start = max(1, mid_start);
            mid_end = min(bb(3), mid_end);

            proj = sum(sub(:, mid_start:mid_end), 1);
            [~, min_idx_local] = min(proj);
            min_idx = mid_start + min_idx_local - 1;

            if min_idx < 1
                min_idx = 1;
            elseif min_idx >= bb(3)
                min_idx = bb(3)-1;
            end

            sub_left = sub(:, 1:min_idx);
            sub_right = sub(:, min_idx+1:end);

            BW_new(y1:y2, x1:x1+min_idx-1) = sub_left;
            BW_new(y1:y2, x1+min_idx:x2) = sub_right;

            new_boxes = [new_boxes;
                x1, y1, size(sub_left,2), size(sub_left,1);
                x1+min_idx, y1, size(sub_right,2), size(sub_right,1)];
        else
            BW_new(y1:y2, x1:x2) = sub;
            new_boxes = [new_boxes; bb];
        end
    end

    num_new = size(new_boxes,1);
    props_new = struct('BoundingBox', cell(num_new,1));
    for i = 1:num_new
        props_new(i).BoundingBox = new_boxes(i,:);
    end
end

function add_row_legend(t5, row, text_str, num_cols)
    ax = nexttile(t5, (row-1)*num_cols + 1);
    if isempty(ax) || ~isvalid(ax)
        error('Invalid axes handle returned by nexttile for row %d', row);
    end
    text(0.5, 0.5, text_str, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', 'FontSize', 10, 'Parent', ax);
    axis(ax, 'off');
end