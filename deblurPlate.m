function results = deblurPlate(input, varargin)
%DEBLURPLATE Deblur, READ and reconstruct a blurred Bangladeshi license plate.
%   results = deblurPlate('data/photo_1_2026-08-26_01-46-56.jpg')
%   results = deblurPlate(img)                      % image array
%   results = deblurPlate('car.jpg', 'ROI', 'manual')  % draw the plate box
%   results = deblurPlate(file, 'Classical', true)  % also show Wiener / TV
%
%   Three results, from one shared network pass:
%     1. Restored       PlateDeblurNet pixel restoration (src/deblurPlateDL.m)
%     2. Reading        PlateReader recognises city, class letter and the six
%                       digits directly from the blurred photo, with a
%                       confidence per field (src/recognizePlateDL.m)
%     3. Reconstructed  a clean, sharp plate typeset from that reading
%                       (src/reconstructPlate.m); low-confidence characters
%                       are drawn in RED so a guess never passes for a fact
%   The full 9-stage pipeline in main.m uses the same networks.
%
%   Name/value options
%     'ROI'        ('full')  'full'   - the file is already a plate crop
%                            'manual' - drag a rectangle round the plate
%                            'auto'   - edge-density guess (unreliable)
%                            [x y w h] - explicit box
%     'Classical'  (false)   also run the old cepstral-PSF + Wiener / TV
%                            restoration for a side-by-side comparison
%     'ShowFigure' (true)
%     'SaveDir'    ('')      default data/<name>_deblurred_DL (only when the
%                            input is a file); set to 'none' to skip saving
%     'ModelPath'  ('')      default models/plateDeblurNet.mat
%     'Recognize'  (true)    run the reader + reconstruction (needs
%                            models/plateReader.mat and plateGlyphs.mat)
%
%   results fields
%     Input, BBox, ROI (color crop), Restored (network resolution, RGB),
%     RestoredFull (same size as the ROI), Info, Reading (see
%     recognizePlateDL), Text, Reconstructed (RGB), and when 'Classical' is on:
%     Wiener, TV, Blur (length/angle estimate).
%
%   Batch over every blurred photo in data/:  runDeepDeblurDemo

p = inputParser;
addParameter(p, 'ROI',        'full');
addParameter(p, 'Classical',  false);
addParameter(p, 'ShowFigure', true);
addParameter(p, 'SaveDir',    '');
addParameter(p, 'ModelPath',  '');
addParameter(p, 'Recognize',  true);
parse(p, varargin{:});
opts = p.Results;

thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'src'));

% ---- read ----------------------------------------------------------------
srcName = 'image';
if ischar(input) || isstring(input)
    f = char(input);
    if exist(f, 'file') ~= 2 && exist(fullfile(thisDir, f), 'file') == 2
        f = fullfile(thisDir, f);
    end
    if exist(f, 'file') ~= 2
        error('deblurPlate:notFound', 'Image not found: %s', char(input));
    end
    raw = imread(f);
    [~, srcName] = fileparts(f);
else
    raw = input;
end
rgb = im2double(raw);
switch size(rgb, 3)
    case 1, rgb = repmat(rgb, 1, 1, 3);
    case 3
    otherwise, rgb = rgb(:, :, 1:3);
end
gray = rgb2gray(rgb);

% ---- ROI -----------------------------------------------------------------
if isnumeric(opts.ROI)
    b = round(opts.ROI);
    x1 = max(1, b(1)); y1 = max(1, b(2));
    x2 = min(size(rgb, 2), b(1) + b(3)); y2 = min(size(rgb, 1), b(2) + b(4));
    bbox = [x1 y1 x2 - x1 y2 - y1];
else
    [~, bbox] = extractROI(gray, 'Method', opts.ROI);
end
x1 = bbox(1); y1 = bbox(2);
x2 = min(size(rgb, 2), x1 + bbox(3)); y2 = min(size(rgb, 1), y1 + bbox(4));
roiRGB = rgb(y1:y2, x1:x2, :);

% ---- deep learning restoration ------------------------------------------
[restored, info] = deblurPlateDL(roiRGB, 'ModelPath', opts.ModelPath, 'Verbose', true);
restoredFull = min(max(imresize(restored, [size(roiRGB, 1) size(roiRGB, 2)], 'bicubic'), 0), 1);

results = struct('Input', rgb, 'BBox', bbox, 'ROI', roiRGB, ...
                 'Restored', restored, 'RestoredFull', restoredFull, 'Info', info);

% ---- recognition + reconstruction ---------------------------------------
readerFile = fullfile(thisDir, 'models', 'plateReader.mat');
doRead = opts.Recognize && exist(readerFile, 'file') == 2;
if doRead
    rd = recognizePlateDL(roiRGB, 'ModelPath', opts.ModelPath);
    results.Reading = rd;
    results.Text = rd.Text;
    results.Reconstructed = reconstructPlate(rd, 'Reference', restored);
    fprintf('Plate reading : %s   /   %s\n', rd.TopLine, rd.BottomLine);
    fprintf('Confidence    : city %.2f, class %.2f, digits %s  (lowest %.2f)\n', ...
        rd.Confidence.city, rd.Confidence.cls, ...
        strjoin(arrayfun(@(k) sprintf('%.2f', rd.Confidence.(sprintf('d%d', k))), ...
                (1 + 2 * strcmp(rd.Format, 'old')):6, 'UniformOutput', false), ' '), rd.MinConfidence);
    if ~isempty(rd.LowConfidenceFields)
        fprintf('Low confidence: %s  (drawn in red on the reconstructed plate)\n', ...
            strjoin(rd.LowConfidenceFields, ', '));
    end
elseif opts.Recognize
    warning('deblurPlate:noReader', 'models/plateReader.mat not found; skipping recognition.');
end

% ---- optional classical baseline ----------------------------------------
if opts.Classical
    roiG = rgb2gray(roiRGB);
    [len, ang] = estimateBlurParameters(roiG, [], 'MinLen', 4, 'MaxLen', 80, ...
        'Refine', false, 'NSR', 0.02, 'Visualize', false, 'Verbose', false);
    psf = generatePSF(len, ang);
    [w, ~, ~, tv] = restoreImage(applyEdgeTaper(roiG, psf), psf, 'NSR', 0.02, ...
        'LucyIter', 8, 'Method', 'both', 'TVWeight', 0.001, 'TVIters', 60);
    results.Wiener = w;
    results.TV     = tv;
    results.Blur   = struct('Length', len, 'Angle', ang);
end

% ---- figure --------------------------------------------------------------
if opts.ShowFigure
    nt = 2 + 2 * opts.Classical + doRead;
    fig = figure('Name', ['Deep-learning deblurring: ' srcName], 'NumberTitle', 'off', ...
                 'Color', 'w', 'Position', [100 100 360 * nt 420]);
    t = tiledlayout(1, nt, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(t, sprintf('%s   |   PlateDeblurNet (%.1f M params, %.2f s)', ...
        strrep(srcName, '_', '\_'), info.NumParams / 1e6, info.Seconds), ...
        'FontWeight', 'bold');
    nexttile; imshow(roiRGB);   title('Blurred input');
    if opts.Classical
        nexttile; imshow(min(max(results.Wiener, 0), 1));
        title(sprintf('Wiener (PSF %g px, %g%s)', results.Blur.Length, results.Blur.Angle, char(176)));
        nexttile; imshow(min(max(results.TV, 0), 1)); title('TV deconvolution');
    end
    nexttile; imshow(restored); title('PlateDeblurNet restoration');
    if doRead
        nexttile; imshow(results.Reconstructed);
        title(sprintf('Reconstructed from reading (min conf %.2f)', results.Reading.MinConfidence));
    end
    results.Figure = fig;
end

% ---- save ----------------------------------------------------------------
saveDir = opts.SaveDir;
if isempty(saveDir) && (ischar(input) || isstring(input))
    saveDir = fullfile(thisDir, 'data', [srcName '_deblurred_DL']);
end
if ~isempty(saveDir) && ~strcmpi(saveDir, 'none')
    if ~exist(saveDir, 'dir'), mkdir(saveDir); end
    imwrite(im2uint8(roiRGB),       fullfile(saveDir, 'roi_blurred.png'));
    imwrite(im2uint8(restored),     fullfile(saveDir, 'restored_deep.png'));
    imwrite(im2uint8(restoredFull), fullfile(saveDir, 'restored_deep_roi_size.png'));
    if doRead
        imwrite(im2uint8(results.Reconstructed), fullfile(saveDir, 'reconstructed_plate.png'));
        fid = fopen(fullfile(saveDir, 'reading.txt'), 'w', 'n', 'UTF-8');
        if fid > 0
            rd = results.Reading;
            fprintf(fid, '%s\n%s\n\nformat %s\n', rd.TopLine, rd.BottomLine, rd.Format);
            fn = fieldnames(rd.Confidence);
            for k = 1:numel(fn), fprintf(fid, 'confidence %-5s %.3f\n', fn{k}, rd.Confidence.(fn{k})); end
            fclose(fid);
        end
    end
    if opts.Classical
        imwrite(im2uint8(min(max(results.Wiener, 0), 1)), fullfile(saveDir, 'restored_wiener.png'));
        imwrite(im2uint8(min(max(results.TV, 0), 1)),     fullfile(saveDir, 'restored_tv.png'));
    end
    if isfield(results, 'Figure') && isgraphics(results.Figure)
        try
            exportgraphics(results.Figure, fullfile(saveDir, 'figure_deep_comparison.png'), 'Resolution', 150);
        catch
            try, print(results.Figure, fullfile(saveDir, 'figure_deep_comparison.png'), '-dpng', '-r150'); catch, end
        end
    end
    results.SaveDir = saveDir;
    fprintf('Saved to %s\n', saveDir);
end
end
