%this code is unnecessary. i made a copy while debugging






%% MAIN - Vehicle License Plate Motion Deblurring Pipeline
% Full pipeline:
%   1. Image acquisition and preprocessing
%   2. Region of interest extraction
%   3. FFT spectrum analysis
%   4. Blur parameter estimation
%   5. PSF generation
%   6. Edge tapering
%   7. Wiener / Lucy-Richardson restoration
%   8. Contrast enhancement and binarization
%   9. OCR and evaluation
%
% Requirements: Image Processing Toolbox (all stages),
%               Computer Vision Toolbox (ocr(), insertText() for the demo)

close all; clc;
addpath('src');

%% ---- CONFIGURATION ----
useSyntheticTest = true;               % true: generate a test image; false: load your own
inputImagePath   = 'data/sample_plate.jpg'; % used only if useSyntheticTest = false
groundTruthText  = 'DHA1234';           % optional; set '' if unknown
roiMethod        = 'auto';              % 'auto' or 'manual'

% --- DEBUG / CALIBRATION ---
% If restoration output looks like noise/garbage rather than a sharper
% plate, the automatically estimated blur length/angle is almost always
% the cause (Wiener/Lucy are very sensitive to PSF mismatch). Set
% useManualBlurParams = true and fill in blurLenManual/blurAngleManual
% to bypass steps 3-4 entirely and test restoration with known-correct
% values. For the synthetic test this should exactly match
% trueLen/trueAngle printed below.
useManualBlurParams = false;
blurLenManual   = 15;
blurAngleManual = 20;

%% ---- STEP 0 (demo only): synthetic blurred plate ----
% Skip this block and set useSyntheticTest = false to run on a real photo.
if useSyntheticTest
    fprintf('Generating synthetic motion-blurred test image...\n');
    canvas = zeros(120, 400, 'uint8');
    canvas = insertText(canvas, [30 30], groundTruthText, ...
        'FontSize', 48, 'TextColor', 'white', 'BoxOpacity', 0);
    sharpGray = im2double(rgb2gray(canvas));

    trueLen = 15; trueAngle = 20;
    imGray = simulateMotionBlur(sharpGray, trueLen, trueAngle, 0.0005);
    imOrigColor = repmat(imGray, [1 1 3]);
    fprintf('  True blur length = %d px, true angle = %d deg\n', trueLen, trueAngle);
else
    fprintf('Loading input image: %s\n', inputImagePath);
    [imGray, imOrigColor] = loadAndPreprocess(inputImagePath, ...
        'ResizeWidth', 800, 'Denoise', true);
end

figure('Name', 'Step 1: Preprocessed Image');
imshow(imGray, []); title('Preprocessed Grayscale Image');

%% ---- STEP 2: ROI Extraction ----
fprintf('Extracting license plate ROI...\n');
[roi, bbox] = extractROI(imGray, 'Method', roiMethod);

figure('Name', 'Step 2: ROI Extraction');
imshow(imGray, []); hold on;
rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
title('Detected License Plate ROI'); hold off;

if useManualBlurParams
    fprintf('Using manual blur parameters (steps 3-4 skipped): len=%d, angle=%.1f\n', ...
        blurLenManual, blurAngleManual);
    blurLen = blurLenManual;
    blurAngle = blurAngleManual;
else
  %% ---- PARAMETER SWEEP ----
% Test lengths 6 to 22 px (step 4) and angles -15 to 15 deg (step 10)
sweepParameters(roi, -15:10:15, 6:4:22);

% Prompt user to pick the best combination from the figure grid
blurLen   = input('Enter best length from grid (e.g. 10): ');
blurAngle = input('Enter best angle from grid (e.g. 0): ');

psf = generatePSF(blurLen, blurAngle);
end

%% ---- STEP 6: Edge Tapering ----
roiTapered = applyEdgeTaper(roi, psf);

%% ---- STEP 7: Restoration (Wiener + Lucy-Richardson) ----
fprintf('Restoring image...\n');
[restoredWiener, restoredLucy] = restoreImage(roiTapered, psf, ...
    'NSR', 0.01, 'LucyIter', 10, 'Method', 'both');
% If restoredLucy looks noisier/worse than restoredWiener, lower
% LucyIter further (e.g. 5) — Richardson-Lucy amplifies noise and PSF
% mismatch much faster than Wiener does.

figure('Name', 'Step 7: Restoration Comparison');
subplot(1, 3, 1); imshow(roi, []); title('Blurred ROI');
subplot(1, 3, 2); imshow(restoredWiener, []); title('Wiener Restoration');
subplot(1, 3, 3); imshow(restoredLucy, []); title('Lucy-Richardson Restoration');

%% ---- STEP 8: Contrast Enhancement & Binarization ----
fprintf('Enhancing and binarizing...\n');
[enhancedW, bwW] = enhanceAndBinarize(restoredWiener, 'Method', 'adaptive');
[enhancedL, bwL] = enhanceAndBinarize(restoredLucy, 'Method', 'adaptive');

figure('Name', 'Step 8: Enhancement & Binarization');
subplot(2, 2, 1); imshow(enhancedW, []); title('Enhanced (Wiener)');
subplot(2, 2, 2); imshow(bwW); title('Binarized (Wiener)');
subplot(2, 2, 3); imshow(enhancedL, []); title('Enhanced (Lucy)');
subplot(2, 2, 4); imshow(bwL); title('Binarized (Lucy)');

%% ---- STEP 9: OCR and Evaluation ----
fprintf('Running OCR...\n');
resultsW = ocrAndEvaluate(bwW, 'GroundTruth', groundTruthText);
resultsL = ocrAndEvaluate(bwL, 'GroundTruth', groundTruthText);

fprintf('\n===== OCR RESULTS (Wiener) =====\n');
disp(resultsW);
fprintf('===== OCR RESULTS (Lucy-Richardson) =====\n');
disp(resultsL);

fprintf('\nPipeline complete.\n');
