function results = main(cfg)
%MAIN Vehicle license plate motion-deblurring pipeline.
%   results = main()          % run with defaults
%   results = main(cfg)       % override any subset of the defaults
%   main                      % run as a script, figures only
%
%   Stages
%     1 Acquisition and preprocessing      loadAndPreprocess
%     2 ROI extraction                     extractROI
%     3 Fourier spectrum analysis          analyzeFFTSpectrum      (cross-check)
%     4 Blur length/angle estimation       estimateBlurParameters  (cepstral)
%     5 PSF generation                     generatePSF
%     6 Edge tapering                      applyEdgeTaper
%     7 Wiener / Lucy-Richardson / TV      restoreImage            (classical baseline)
%     7b Deep-learning restoration         deblurPlateDL           (PlateDeblurNet, DEFAULT)
%     7c Plate reading + reconstruction    recognizePlateDL        (PlateReader)
%                                          + reconstructPlate      (clean plate from the reading)
%     8 Enhancement and binarization       enhanceAndBinarize
%     9 OCR and evaluation                 ocrAndEvaluate
%       or, for Bengali plates              segmentBanglaPlate
%                                           + banglaTemplates
%                                           + recognizeBanglaPlate
%
%   CONFIGURATION (cfg fields, all optional)
%     UseSyntheticTest (false)  true generates a known blurred plate instead
%                               of reading a file, so length/angle recovery
%                               can be scored against ground truth
%     InputImagePath   ('data/blur.png')
%     GroundTruthText  ('')     only meaningful for Latin/numeric plates
%     Script           ('english') 'english' routes stage 9 to MATLAB's ocr;
%                               'bangla' routes it to template matching,
%                               because the ocr engine here is English with
%                               an A-Z0-9 set and cannot read Bengali at all
%     BanglaTemplateSource ('charset') where the Bengali templates come from
%                               when BanglaRefImage is empty. 'charset' builds
%                               the whole plate alphabet from the two glyph
%                               charts in data/, so 'Script','bangla' needs no
%                               further configuration at all.
%     BanglaInclude    ('plate') which classes the charset contains:
%                               'plate' | 'all' | 'letters' | 'digits'.
%                               'plate' is 'all' minus the three combining
%                               marks, which never stand alone on a plate and
%                               which normalise into something that collides
%                               with the digit one. See banglaCharset.
%     BanglaRefImage   ('')     OPTIONAL override: a SHARP reference image to
%                               cut templates from instead. Narrower than the
%                               charset (only the characters on that one
%                               plate) but in a real plate typeface rather
%                               than a chart typeface, so it matches better
%                               where it applies. Needs BanglaRefLabels.
%     BanglaRefLabels  ({})     one label per character of the reference,
%                               left to right, e.g. {'DHA','AA','KA','AA'}.
%                               Repeat a label when a glyph repeats.
%     BanglaGroundTruth ({})    expected labels for the input, for scoring
%     ROIMethod        ('full') 'full' | 'auto' | 'manual'
%                               'manual' opens a window and you drag the box
%                               yourself -- use this whenever auto guesses
%                               wrong, which it will on a cluttered photo.
%                               'full' is right only when the file is already
%                               a cropped plate.
%     NSR              (0.02)   Wiener regularisation
%     LucyIter         (8)
%     RestoreMethod    ('deep') which restoration feeds recognition, the
%                               before/after figure and the saved crop:
%                               'deep' | 'tv' | 'wiener' | 'lucy'.
%                               'deep' is the trained convolutional network
%                               PlateDeblurNet (src/deblurPlateDL.m). It needs
%                               no PSF estimate, so it is not derailed by
%                               real, non-linear camera shake the way the
%                               deconvolution filters are, and it learned
%                               the shapes of Bangla glyphs from ~70k
%                               training plates. The classical three are
%                               still computed and shown as a baseline; this
%                               only picks which one is "the result". If the
%                               model file is missing, 'deep' falls back to
%                               'tv' with a warning.
%     DeepModelPath    ('')     default models/plateDeblurNet.mat
%     ReadPlate        (true)   stage 7c: read city / class / digits with the
%                               PlateReader network and typeset a clean plate
%                               from the reading (needs models/plateReader.mat)
%     TVWeight         (0.001)  TV regularisation weight. Larger = flatter,
%                               smaller = sharper but noisier.
%     TVIters          (60)     ADMM iterations for the TV solve.
%     MinLen/MaxLen    (4/80)   blur-length search range, px
%     Refine           (false)  bounded refinement of the estimate. Off
%                               because it measurably degrades the cepstral
%                               estimate; see estimateBlurParameters.m
%     UseManualBlurParams (false), BlurLenManual (52), BlurAngleManual (0)
%     ShowFigures      (true), ShowDiagnostics (true), ShowSweep (true)
%     SaveOutputs      (true), OutputDir ('' -> data/<name>_deblurred)
%     ReferenceImagePath ('')   sharp reference; enables PSNR/SSIM scoring
%     Verbose          (true)
%
%   RETURNS a struct with the images, the estimated parameters, the quality
%   metrics and the OCR results, so the pipeline can be scripted or swept
%   without reading anything off a figure.
%
%   Example: score the estimator against ground truth
%     r = main(struct('UseSyntheticTest', true));
%     r.Blur
%
%   Example: run on a real photo of a whole car
%     r = main(struct('InputImagePath','data/img2.png','ROIMethod','auto'));
%
%   Example: pick the plate yourself when auto gets it wrong
%     r = main(struct('InputImagePath','data/img4.png','ROIMethod','manual'));
%
%   Example: read a Bengali plate with no extra configuration
%     r = main(struct('InputImagePath','data/blur.png','Script','bangla'));
%     r.Bangla.Text          % ASCII labels, e.g. 'DHA AA KA AA'
%     r.Bangla.TextUnicode   % the same reading as actual Bengali
%
%   Example: use a sharp reference plate as the template source instead
%     r = main(struct( ...
%         'InputImagePath',   'data/blur.png', ...
%         'Script',           'bangla', ...
%         'BanglaRefImage',   'data/dhaka_l.png', ...
%         'BanglaRefLabels',  {{'DHA','AA','KA','AA'}}, ...
%         'BanglaGroundTruth',{{'DHA','AA','KA','AA'}}));
%     r.Bangla.Text
%   Note the DOUBLE braces: struct() consumes one level of cell nesting.

if nargin < 1 || isempty(cfg)
    cfg = struct();
end
cfg = mergeConfig(cfg);

thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'src'));

cfg = checkCapabilities(cfg);

if cfg.ShowFigures
    close all;
end
results = struct();
results.Config = cfg;

vprintf(cfg, '\n================ LICENSE PLATE DEBLURRING ================\n');

%% ---- STEP 1: acquisition ------------------------------------------------
if cfg.UseSyntheticTest
    vprintf(cfg, 'Step 1  synthetic test plate ("%s")\n', cfg.SyntheticText);
    [sharpGray, plateInfo] = makeSyntheticPlate(cfg.SyntheticText);

    % Keep the simulated blur shorter than the character height, otherwise
    % the glyphs are destroyed rather than merely smeared and no restoration
    % can recover them. (The old demo used 55 px on 48 px text.)
    trueLen   = min(cfg.SyntheticLen, round(0.55 * plateInfo.CharHeight));
    trueAngle = cfg.SyntheticAngle;

    imGray = simulateMotionBlur(sharpGray, trueLen, trueAngle, cfg.SyntheticNoise);
    srcName = 'synthetic';
    loadInfo = struct('Polarity', detectTextPolarity(imGray), 'Scale', 1);

    imColor = repmat(imGray, 1, 1, 3);

    results.Truth = struct('Length', trueLen, 'Angle', trueAngle, ...
                           'Text', plateInfo.Text, 'Sharp', sharpGray);
    referenceImage = sharpGray;
    groundTruthText = plateInfo.Text;

    vprintf(cfg, '        true blur: %d px at %g deg (text height %d px)\n', ...
        trueLen, trueAngle, plateInfo.CharHeight);
    if trueLen < cfg.SyntheticLen
        vprintf(cfg, '        (requested %d px was clamped to keep the text legible)\n', ...
            cfg.SyntheticLen);
    end
else
    imgPath = resolvePath(cfg.InputImagePath, thisDir);
    vprintf(cfg, 'Step 1  loading %s\n', cfg.InputImagePath);
    [imGray, imColor, loadInfo] = loadAndPreprocess(imgPath, ...
        'MaxWidth', cfg.MaxWidth, 'Denoise', cfg.Denoise);
    % The network wants the colour image, without the median denoise.
    imColor = im2double(imColor);
    if size(imColor, 3) == 1, imColor = repmat(imColor, 1, 1, 3); end
    [~, srcName] = fileparts(imgPath);

    referenceImage = [];
    if ~isempty(cfg.ReferenceImagePath)
        refPath = resolvePath(cfg.ReferenceImagePath, thisDir);
        referenceImage = loadAndPreprocess(refPath, ...
            'MaxWidth', cfg.MaxWidth, 'Denoise', false);
    end
    groundTruthText = cfg.GroundTruthText;

    vprintf(cfg, '        %dx%d, text polarity: %s%s\n', ...
        size(imGray, 2), size(imGray, 1), loadInfo.Polarity, ...
        ternary(isfield(loadInfo, 'HadAlpha') && loadInfo.HadAlpha, ...
                ' (alpha channel dropped)', ''));
end

results.Input    = imGray;
results.Polarity = loadInfo.Polarity;

%% ---- STEP 2: ROI -------------------------------------------------------
[roi, bbox, roiInfo] = extractROI(imGray, 'Method', cfg.ROIMethod, ...
    'Polarity', loadInfo.Polarity);
vprintf(cfg, 'Step 2  ROI %dx%d via "%s"\n', ...
    size(roi, 2), size(roi, 1), roiInfo.Method);

results.ROI     = roi;
results.BBox    = bbox;
% Same box on the colour image, for the deep-learning restorer (stage 7b).
rx1 = max(1, round(bbox(1))); ry1 = max(1, round(bbox(2)));
rx2 = min(size(imColor, 2), rx1 + round(bbox(3)));
ry2 = min(size(imColor, 1), ry1 + round(bbox(4)));
roiColor = imColor(ry1:ry2, rx1:rx2, :);
if ~isequal([size(roiColor, 1) size(roiColor, 2)], size(roi))
    roiColor = imresize(roiColor, size(roi));
end
results.ROIColor = roiColor;
results.ROIInfo = roiInfo;

%% ---- STEPS 3-4: blur parameters ---------------------------------------
if cfg.UseManualBlurParams
    blurLen   = cfg.BlurLenManual;
    blurAngle = cfg.BlurAngleManual;
    blurInfo  = struct('Manual', true, 'Confidence', NaN);
    vprintf(cfg, 'Steps 3-4  manual override: %g px at %g deg\n', blurLen, blurAngle);
else
    angleHint = [];
    if cfg.ShowDiagnostics || cfg.UseFFTCrossCheck
        [~, angleHint] = analyzeFFTSpectrum(roi, 'Visualize', cfg.ShowDiagnostics);
    end

    vprintf(cfg, 'Steps 3-4  estimating blur parameters\n');
    [blurLen, blurAngle, blurInfo] = estimateBlurParameters(roi, angleHint, ...
        'MinLen', cfg.MinLen, 'MaxLen', cfg.MaxLen, ...
        'Refine', cfg.Refine, 'NSR', cfg.NSR, ...
        'Visualize', cfg.ShowDiagnostics, 'Verbose', cfg.Verbose);
    blurInfo.Manual = false;
end

results.Blur = struct('Length', blurLen, 'Angle', blurAngle, 'Info', blurInfo);

vprintf(cfg, '        --> blur length %g px, angle %g deg\n', blurLen, blurAngle);
if cfg.UseSyntheticTest
    vprintf(cfg, '        vs truth : length error %+.1f px, angle error %+.1f deg\n', ...
        blurLen - results.Truth.Length, ...
        wrapAngleError(blurAngle - results.Truth.Angle));
    results.Truth.LengthError = blurLen - results.Truth.Length;
    results.Truth.AngleError  = wrapAngleError(blurAngle - results.Truth.Angle);
end

%% ---- STEP 5: PSF -------------------------------------------------------
psf = generatePSF(blurLen, blurAngle);
results.PSF = psf;

%% ---- STEP 6: edge tapering --------------------------------------------
[roiTapered, taperInfo] = applyEdgeTaper(roi, psf);
results.TaperInfo = taperInfo;

%% ---- STEP 7: restoration ----------------------------------------------
vprintf(cfg, 'Step 7  restoring (NSR %.3g, Lucy %d it, TV lambda %.4g / %d it)\n', ...
    cfg.NSR, cfg.LucyIter, cfg.TVWeight, cfg.TVIters);
[restoredWiener, restoredLucy, restInfo, restoredTV] = restoreImage(roiTapered, psf, ...
    'NSR', cfg.NSR, 'LucyIter', cfg.LucyIter, 'Method', 'both', ...
    'TVWeight', cfg.TVWeight, 'TVIters', cfg.TVIters);

results.RestoredWiener = restoredWiener;
results.RestoredLucy   = restoredLucy;
results.RestoredTV     = restoredTV;
results.RestoreInfo    = restInfo;

vprintf(cfg, ['        ringing indicator (pixels clipped): ' ...
              'Wiener %.1f%%, Lucy %.1f%%, TV %.1f%%\n'], ...
    100 * restInfo.WienerOvershoot, 100 * restInfo.LucyOvershoot, ...
    100 * restInfo.TVOvershoot);

%% ---- STEP 7b: deep-learning restoration -------------------------------
restoredDeep = []; restoredDeepRGB = [];
if cfg.DeepAvailable
    vprintf(cfg, 'Step 7b deep-learning restoration (PlateDeblurNet)\n');
    [restoredDeepRGB, deepInfo] = deblurPlateDL(roiColor, 'ModelPath', cfg.DeepModelPath);
    % Grey copy at ROI size so stages 8-9 and the metrics compare like with like.
    restoredDeep = min(max(rgb2gray(imresize(restoredDeepRGB, size(roi), 'bicubic')), 0), 1);
    results.DeepInfo = deepInfo;
    vprintf(cfg, '        network input %dx%d -> output %dx%d in %.2f s\n', ...
        deepInfo.InputSize(2), deepInfo.InputSize(1), ...
        deepInfo.OutputSize(2), deepInfo.OutputSize(1), deepInfo.Seconds);
end
%% ---- STEP 7c: read the plate and reconstruct it ------------------------
results.PlateReading = []; results.Reconstructed = [];
readerPath = fullfile(thisDir, 'models', 'plateReader.mat');
if cfg.DeepAvailable && cfg.ReadPlate && exist(readerPath, 'file') == 2
    vprintf(cfg, 'Step 7c reading the plate (PlateReader)\n');
    rd = recognizePlateDL(roiColor, 'ModelPath', cfg.DeepModelPath);
    results.PlateReading = rd;
    results.Reconstructed = reconstructPlate(rd, 'Reference', restoredDeepRGB);
    vprintf(cfg, '        read "%s / %s" (lowest field confidence %.2f)\n', ...
        rd.TopLine, rd.BottomLine, rd.MinConfidence);
end
results.RestoredDeep    = restoredDeep;      % grey, ROI size
results.RestoredDeepRGB = restoredDeepRGB;   % colour, network resolution (2x input)

% Choose which restoration drives enhancement, recognition, the before/after
% figure and the saved crop. TV is the default: on real handheld shake it is
% visibly the clearest, because it reconstructs solid strokes instead of the
% ringing a Wiener filter leaves once the PSF is imperfect. 'wiener'/'lucy'
% stay selectable. The synthetic PSNR benchmark below is always measured on
% Wiener and Lucy, so it is unaffected by this choice.
[restoredMain, methodUsed] = pickRestoration(cfg.RestoreMethod, ...
    restoredWiener, restoredLucy, restoredTV, restoredDeep);
results.RestoredMain      = restoredMain;
results.RestoreMethodUsed = methodUsed;
vprintf(cfg, '        primary restoration for stages 8-9: %s\n', methodUsed);

%% ---- STEP 8: enhancement and binarization -----------------------------
vprintf(cfg, 'Step 8  enhancing and binarizing (polarity: %s)\n', loadInfo.Polarity);
[enhancedW, bwW, binInfoW] = enhanceAndBinarize(restoredWiener, ...
    'Polarity', loadInfo.Polarity, 'Method', cfg.BinarizeMethod);
[enhancedL, bwL, binInfoL] = enhanceAndBinarize(restoredLucy, ...
    'Polarity', loadInfo.Polarity, 'Method', cfg.BinarizeMethod);
% The chosen primary restoration, enhanced the same way; this is what
% recognition, the before/after figure and the saved crop use.
[enhancedMain, bwMain, binInfoMain] = enhanceAndBinarize(restoredMain, ...
    'Polarity', loadInfo.Polarity, 'Method', cfg.BinarizeMethod);

results.EnhancedWiener  = enhancedW;
results.EnhancedLucy    = enhancedL;
results.EnhancedMain    = enhancedMain;
results.BinaryWiener    = bwW;
results.BinaryLucy      = bwL;
results.BinaryMain      = bwMain;
results.BinarizeInfo    = struct('Wiener', binInfoW, 'Lucy', binInfoL, ...
                                 'Main', binInfoMain);

%% ---- Quality metrics ---------------------------------------------------
results.Metrics = computeMetrics(roi, restoredWiener, restoredLucy, referenceImage, restoredDeep, restoredTV);
if cfg.Verbose
    printMetrics(results.Metrics);
end

%% ---- STEP 9: OCR ------------------------------------------------------
if cfg.RunOCR
    switch lower(cfg.Script)
        case 'bangla'
            vprintf(cfg, 'Step 9  Bangla recognition (template matching)\n');
            results.Bangla = runBangla(enhancedMain, cfg);
        otherwise
            vprintf(cfg, 'Step 9  OCR\n');
            results.OCRWiener = ocrAndEvaluate(bwW, 'GroundTruth', groundTruthText, ...
                'Grayscale', enhancedW, 'Verbose', cfg.Verbose);
            results.OCRLucy = ocrAndEvaluate(bwL, 'GroundTruth', groundTruthText, ...
                'Grayscale', enhancedL, 'Verbose', cfg.Verbose);
            if cfg.Verbose
                printOCR(results.OCRWiener, results.OCRLucy, groundTruthText);
            end
    end
else
    vprintf(cfg, 'Step 9  OCR skipped (RunOCR = false)\n');
end

%% ---- Figures ----------------------------------------------------------
if cfg.ShowFigures
    results.Figures = drawFigures(results, cfg, srcName);
    if cfg.ShowSweep && ~cfg.UseManualBlurParams
        sweepAround(roi, blurLen, blurAngle, cfg);
    end
end

%% ---- Save -------------------------------------------------------------
if cfg.SaveOutputs
    outDir = cfg.OutputDir;
    if isempty(outDir)
        outDir = fullfile(thisDir, 'data', [srcName '_deblurred']);
    end
    results.OutputDir = saveOutputs(outDir, results, cfg);
    vprintf(cfg, '\nSaved outputs to %s\n', results.OutputDir);
end

vprintf(cfg, '\nPipeline complete.\n');
vprintf(cfg, '=========================================================\n\n');

end

% =======================================================================
% Configuration
% =======================================================================
function cfg = mergeConfig(user)
d = struct( ...
    'UseSyntheticTest',    false, ...
    'InputImagePath',      'data/blur.png', ...
    'ReferenceImagePath',  '', ...
    'GroundTruthText',     '', ...
    'ROIMethod',           'full', ...
    'MaxWidth',            900, ...
    'Denoise',             true, ...
    'NSR',                 0.02, ...
    'LucyIter',            8, ...
    'RestoreMethod',       'deep', ...
    'DeepModelPath',       '', ...
    'ReadPlate',           true, ...
    'TVWeight',            0.001, ...
    'TVIters',             60, ...
    'MinLen',              4, ...
    'MaxLen',              80, ...
    'Refine',              false, ...
    'UseFFTCrossCheck',    true, ...
    'BinarizeMethod',      'adaptive', ...
    'UseManualBlurParams', false, ...
    'BlurLenManual',       52, ...
    'BlurAngleManual',     0, ...
    'SyntheticText',       'DHA1234', ...
    'SyntheticLen',        21, ...
    'SyntheticAngle',      12, ...
    'SyntheticNoise',      0.0005, ...
    'RunOCR',              true, ...
    'Script',              'english', ...
    'BanglaTemplateSource', 'charset', ...
    'BanglaInclude',       'plate', ...
    'BanglaRefImage',      '', ...
    'BanglaRefLabels',     {{}}, ...
    'BanglaGroundTruth',   {{}}, ...
    'ShowFigures',         true, ...
    'ShowDiagnostics',     true, ...
    'ShowSweep',           true, ...
    'SaveOutputs',         true, ...
    'OutputDir',           '', ...
    'Verbose',             true);

cfg = d;
fn = fieldnames(user);
for k = 1:numel(fn)
    if ~isfield(d, fn{k})
        warning('main:unknownOption', 'Ignoring unknown cfg field "%s".', fn{k});
        continue;
    end
    cfg.(fn{k}) = user.(fn{k});
end
end

% =======================================================================
function [img, name] = pickRestoration(method, w, l, tv, deep)
%PICKRESTORATION Select the restoration that feeds stages 8-9 and display.
%   Keeps the mapping from cfg.RestoreMethod to an image in one place so the
%   figure, the recognition path and the saved crop can never disagree about
%   which restoration "the result" refers to.
switch lower(method)
    case {'deep', 'dl', 'cnn'}
        if isempty(deep)
            img = tv; name = 'TV (total variation)';
        else
            img = deep; name = 'Deep learning (PlateDeblurNet)';
        end
    case 'wiener'
        img = w;  name = 'Wiener';
    case 'lucy'
        img = l;  name = 'Lucy-Richardson';
    case {'tv', 'total-variation', 'totalvariation'}
        img = tv; name = 'TV (total variation)';
    otherwise
        warning('main:badRestoreMethod', ...
            ['RestoreMethod "%s" is not recognised (use ''tv'', ''wiener'' ', ...
             'or ''lucy'' or ''deep''). Falling back to TV.'], method);
        img = tv; name = 'TV (total variation)';
end
end

% =======================================================================
% Metrics
% =======================================================================
function m = computeMetrics(roi, wiener, lucy, reference, deep, tv)
m = struct();
m.EdgeEnergyBlurred = edgeEnergy(roi);
m.HasDeep = ~isempty(deep);
m.EdgeEnergyWiener  = edgeEnergy(wiener);
m.EdgeEnergyLucy    = edgeEnergy(lucy);
m.SharpnessGainWiener = m.EdgeEnergyWiener / max(m.EdgeEnergyBlurred, eps);
m.SharpnessGainLucy   = m.EdgeEnergyLucy   / max(m.EdgeEnergyBlurred, eps);
if m.HasDeep
    m.EdgeEnergyDeep    = edgeEnergy(deep);
    m.SharpnessGainDeep = m.EdgeEnergyDeep / max(m.EdgeEnergyBlurred, eps);
end

m.HasReference = false;
if ~isempty(reference)
    % Compare all four images on the SAME scale -- no per-image min-to-max
    % normalisation, which would flatter or penalise them inconsistently.
    ref = clip01(reference);
    cur = clip01(roi);
    if isequal(size(ref), size(cur))
        m.HasReference = true;
        w = clip01(wiener);
        l = clip01(lucy);
        m.PSNRBlurred = psnr(cur, ref);
        m.PSNRWiener  = psnr(w, ref);
        m.PSNRLucy    = psnr(l, ref);
        m.PSNRTV      = psnr(clip01(tv), ref);
        if m.HasDeep, m.PSNRDeep = psnr(clip01(deep), ref); end
        try
            m.SSIMBlurred = ssim(cur, ref);
            m.SSIMWiener  = ssim(w, ref);
            m.SSIMLucy    = ssim(l, ref);
            m.SSIMTV      = ssim(clip01(tv), ref);
            if m.HasDeep, m.SSIMDeep = ssim(clip01(deep), ref); end
        catch
            m.SSIMBlurred = NaN; m.SSIMWiener = NaN; m.SSIMLucy = NaN;
            m.SSIMTV = NaN; m.SSIMDeep = NaN;
        end
    end
end
end

function y = clip01(x)
y = min(max(im2double(x), 0), 1);
end

function e = edgeEnergy(im)
im = clip01(im);
m = 8;
if size(im, 1) > 3 * m && size(im, 2) > 3 * m
    im = im(m + 1:end - m, m + 1:end - m);
end
[gx, gy] = imgradientxy(im, 'sobel');
e = mean(gx(:) .^ 2 + gy(:) .^ 2);
end

function printMetrics(m)
fprintf('\n  Restoration quality\n');
if m.HasDeep
    fprintf('  deep learning          sharpness gain %.2fx', m.SharpnessGainDeep);
    if m.HasReference
        fprintf(', PSNR %.2f dB', m.PSNRDeep);
        if isfield(m, 'SSIMDeep') && ~isnan(m.SSIMDeep), fprintf(', SSIM %.3f', m.SSIMDeep); end
    end
    fprintf('\n');
end
fprintf('  %-22s %12s %12s\n', '', 'Wiener', 'Lucy');
fprintf('  %-22s %12.2fx %11.2fx\n', 'sharpness gain', ...
    m.SharpnessGainWiener, m.SharpnessGainLucy);
if m.HasReference
    fprintf('  %-22s %12.2f %12.2f   (blurred %.2f)\n', 'PSNR vs reference dB', ...
        m.PSNRWiener, m.PSNRLucy, m.PSNRBlurred);
    if ~isnan(m.SSIMWiener)
        fprintf('  %-22s %12.3f %12.3f   (blurred %.3f)\n', 'SSIM vs reference', ...
            m.SSIMWiener, m.SSIMLucy, m.SSIMBlurred);
    end
end
end

function printOCR(rw, rl, gt)
fprintf('\n  OCR results\n');
fprintf('  %-10s %-14s %-22s %10s\n', 'method', 'text', 'variant', 'confidence');
fprintf('  %-10s %-14s %-22s %10.2f\n', 'Wiener', quoted(rw.RecognizedText), ...
    rw.Variant, rw.MeanConfidence);
fprintf('  %-10s %-14s %-22s %10.2f\n', 'Lucy', quoted(rl.RecognizedText), ...
    rl.Variant, rl.MeanConfidence);
if ~isempty(gt) && isfield(rw, 'CharacterAccuracy')
    fprintf('  ground truth "%s": Wiener %.0f%% (edit %d)%s, Lucy %.0f%% (edit %d)%s\n', ...
        rw.GroundTruth, ...
        100 * rw.CharacterAccuracy, rw.EditDistance, exactTag(rw.ExactMatch), ...
        100 * rl.CharacterAccuracy, rl.EditDistance, exactTag(rl.ExactMatch));
else
    fprintf(['  (no Latin/numeric ground truth set, so no accuracy score. Note the\n' ...
             '   images in data/ contain Bengali script, which this OCR engine\n' ...
             '   cannot read -- judge those restorations visually.)\n']);
end
end

function s = exactTag(tf)
if tf, s = ' EXACT'; else, s = ''; end
end

function s = quoted(t)
s = ['"' t '"'];
end

% =======================================================================
% Figures
% =======================================================================
function figs = drawFigures(r, cfg, srcName)
figs = struct();
% NOTE: these use imshow(x) and never imshow(x, []). Every pipeline image is
% already a double in [0,1], and the empty-limits form would rescale each
% panel min-to-max independently -- reintroducing exactly the contrast loss
% that restoreImage.m now avoids, and making panels incomparable.

% ---- Overview ---------------------------------------------------------
figs.Overview = figure('Name', 'Pipeline overview', 'NumberTitle', 'off', ...
    'Color', 'w', 'Position', [80 80 1180 680]);
t = tiledlayout(2, 5, 'Padding', 'compact', 'TileSpacing', 'compact');
title(t, sprintf('%s   |   blur %g px at %g deg   |   NSR %.3g', ...
    strrep(srcName, '_', '\_'), r.Blur.Length, r.Blur.Angle, cfg.NSR), ...
    'FontWeight', 'bold', 'FontSize', 12);

nexttile; imshow(r.Input);  title('1. Input (preprocessed)');
nexttile;
imshow(r.Input); hold on;
rectangle('Position', r.BBox, 'EdgeColor', [1 0.2 0.2], 'LineWidth', 2);
hold off; title(sprintf('2. ROI (%s)', strrep(r.ROIInfo.Method, '_', '\_')));

nexttile;
% The PSF is a kernel whose values are tiny, so here [] is the right choice.
imshow(r.PSF, [], 'InitialMagnification', 'fit');
axis on; set(gca, 'XTick', [], 'YTick', []);
title(sprintf('5. PSF (%g px, %g%s)', r.Blur.Length, r.Blur.Angle, char(176)));

nexttile; imshow(r.RestoredWiener); title('7. Wiener restoration');
nexttile; imshow(r.RestoredTV);     title('7. TV restoration');
nexttile;
if ~isempty(r.RestoredDeepRGB)
    imshow(r.RestoredDeepRGB); title('7b. Deep learning (PlateDeblurNet)');
else
    axis off; title('7b. Deep learning: model not found');
end
nexttile;
if ~isempty(r.Reconstructed)
    imshow(r.Reconstructed);
    title(sprintf('7c. Reconstructed from reading (conf %.2f)', r.PlateReading.MinConfidence));
else
    axis off; title('7c. Plate reading: off / model not found');
end
nexttile; imshow(r.BinaryMain);     title(sprintf('8. Binarized (%s)', ...
    strrep(r.RestoreMethodUsed, '_', '\_')));

% ---- Before / after ---------------------------------------------------
figs.Comparison = figure('Name', 'Before and after', 'NumberTitle', 'off', ...
    'Color', 'w', 'Position', [120 120 1100 520]);
t2 = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
title(t2, 'Blurred ROI  vs  restored', 'FontWeight', 'bold', 'FontSize', 12);
nexttile; imshow(r.ROIColor); title('Before');
nexttile;
if ~isempty(r.Reconstructed)
    imshow(r.Reconstructed);
elseif strncmp(r.RestoreMethodUsed, 'Deep', 4) && ~isempty(r.RestoredDeepRGB)
    imshow(r.RestoredDeepRGB);
else
    imshow(r.RestoredMain);
end
gainMain = edgeEnergy(r.RestoredMain) / max(edgeEnergy(r.ROI), eps);
if ~isempty(r.Reconstructed)
    title(sprintf('After: reconstructed from the network reading (lowest confidence %.2f)', ...
        r.PlateReading.MinConfidence));
else
    title(sprintf('After: %s (sharpness x%.2f)', ...
        strrep(r.RestoreMethodUsed, '_', '\_'), gainMain));
end

% ---- Enhancement / binarization --------------------------------------
figs.Binarize = figure('Name', 'Enhancement and binarization', ...
    'NumberTitle', 'off', 'Color', 'w', 'Position', [160 160 1100 620]);
t3 = tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
title(t3, sprintf('Stage 8 (detected text polarity: %s)', r.Polarity), ...
    'FontWeight', 'bold', 'FontSize', 12);
nexttile; imshow(r.EnhancedWiener); title('Enhanced (Wiener)');
nexttile; imshow(r.BinaryWiener);   title('Binarized (Wiener)');
nexttile; imshow(r.EnhancedLucy);   title('Enhanced (Lucy)');
nexttile; imshow(r.BinaryLucy);     title('Binarized (Lucy)');

% ---- Synthetic ground-truth panel ------------------------------------
if isfield(r, 'Truth')
    figs.Truth = figure('Name', 'Synthetic ground truth', ...
        'NumberTitle', 'off', 'Color', 'w', 'Position', [200 200 1100 480]);
    t4 = tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(t4, sprintf(['Ground truth %g px at %g%s   |   estimated %g px at %g%s' ...
        '   |   error %+.1f px, %+.1f%s'], ...
        r.Truth.Length, r.Truth.Angle, char(176), ...
        r.Blur.Length, r.Blur.Angle, char(176), ...
        r.Truth.LengthError, r.Truth.AngleError, char(176)), ...
        'FontWeight', 'bold', 'FontSize', 12);
    nexttile; imshow(r.Truth.Sharp);     title('Original (sharp)');
    nexttile; imshow(r.ROI);             title('Blurred');
    nexttile; imshow(r.RestoredMain);    title(sprintf('Restored (%s)', ...
        strrep(r.RestoreMethodUsed, '_', '\_')));
end
end

% =======================================================================
function sweepAround(roi, blurLen, blurAngle, cfg)
%SWEEPAROUND Confirmation grid centred on the chosen parameters.
%   The point of the spread is to let you see that the estimate really is the
%   local optimum: shorter lengths leave the text soft, longer ones fill the
%   background with criss-cross ringing. The estimator's pick is outlined in
%   blue, the highest-scoring cell in orange, and green means they agree --
%   be aware that the score is only edge energy, which drifts long, so on
%   real photographs they often do not agree and your eyes win.
lens = round(blurLen * [0.6 0.8 1.0 1.25]);
lens = unique(max(cfg.MinLen, lens));
angs = round(blurAngle + [-8 0 8]);
try
    sweepParameters(roi, angs, lens, 'NSR', cfg.NSR, ...
        'Highlight', [blurLen blurAngle], 'Visualize', true);
catch ME
    warning('main:sweepFailed', 'Sweep figure skipped: %s', ME.message);
end
end

% =======================================================================
function cfg = checkCapabilities(cfg)
%CHECKCAPABILITIES Verify the toolboxes each stage needs, and degrade.
%   Called once, before any image is touched, so a missing toolbox produces
%   one clear sentence here rather than an obscure "Undefined function"
%   twenty lines into stage 6.
%
%   WHAT THIS PIPELINE ACTUALLY NEEDS
%     Image Processing Toolbox     REQUIRED. Everything from imbinarize to
%                                  deconvwnr lives there; there is no
%                                  meaningful fallback and we error out.
%     Computer Vision Toolbox      only for ocr(), i.e. Script 'english'.
%                                  Missing it turns OCR off and leaves the
%                                  restoration stages fully working.
%     Deep Learning Toolbox        NOT NEEDED, deliberately. The neural
%                                  network of stage 7b (deblurPlateDL) is
%                                  evaluated with plain matrix products, and
%                                  the Bengali reader is segmentation plus
%                                  normalised cross-correlation. Both run on
%                                  any install.
if ~exist('imbinarize', 'file') || ~exist('deconvwnr', 'file')
    error('main:noImageProcessingToolbox', ...
        ['This pipeline requires the Image Processing Toolbox ', ...
         '(imbinarize, deconvwnr, fspecial, regionprops). ', ...
         'Check with: ver(''images'')']);
end

% Deep-learning restorer: plain MATLAB, no toolbox, but it needs its weights.
modelPath = cfg.DeepModelPath;
if isempty(modelPath)
    modelPath = fullfile(fileparts(mfilename('fullpath')), 'models', 'plateDeblurNet.mat');
end
cfg.DeepModelPath = modelPath;
cfg.DeepAvailable = exist(modelPath, 'file') == 2 && exist('deblurPlateDL', 'file') == 2;
if ~cfg.DeepAvailable && any(strcmpi(cfg.RestoreMethod, {'deep', 'dl', 'cnn'}))
    warning('main:noDeepModel', ...
        ['Deep-learning model not found (%s). Stage 7b is skipped and ', ...
         'RestoreMethod falls back to ''tv''.'], modelPath);
end

isBangla = strcmpi(cfg.Script, 'bangla');

if cfg.RunOCR && ~isBangla && ~exist('ocr', 'file')
    warning('main:noOCR', ...
        ['ocr() is unavailable (it needs the Computer Vision Toolbox), so ', ...
         'stage 9 is skipped. Stages 1-8 are unaffected and the restored ', ...
         'images are still saved. For Bengali plates set ', ...
         '''Script'',''bangla'', which does not use ocr() at all.']);
    cfg.RunOCR = false;
end

% When Bangla will build its own templates from the shipped charts, check the
% charts are actually there now rather than at stage 9, after minutes of
% deconvolution. Only a warning: stages 1-8 still produce saveable output.
if isBangla && isempty(cfg.BanglaRefImage) && ...
        strcmpi(cfg.BanglaTemplateSource, 'charset')
    thisDir = fileparts(mfilename('fullpath'));
    cs = banglaCharset('Include', cfg.BanglaInclude);
    for f = {cs.LetterChart, cs.NumberChart}
        rel = f{1};
        if ~exist(rel, 'file') && ~exist(fullfile(thisDir, rel), 'file')
            warning('main:chartMissing', ...
                ['Bangla template chart "%s" was not found. Restore it, or ', ...
                 'set BanglaRefImage to a sharp reference plate instead.'], rel);
        end
    end
end
end

% =======================================================================
function out = runBangla(enhanced, cfg)
%RUNBANGLA Stage 9 for Bengali plates: segment, then match templates.
%   Deliberately warns instead of erroring. By this point stages 1-8 have
%   already produced restored images worth saving, and aborting here would
%   skip saveOutputs and throw that work away.
%
%   Two template routes, chosen automatically:
%
%     BanglaRefImage empty (the default)  Build the whole plate alphabet from
%         the two glyph charts in data/ via banglaTemplates 'charset'. No
%         configuration at all beyond 'Script','bangla'. 57 classes, measured
%         100% clean and at 9 px blur, 88% at 55 px, digits 10/10 at every
%         level tested -- see banglaCharset for the full table.
%
%     BanglaRefImage set                  Cut glyphs from that one sharp
%         reference plate instead. Narrower (only the characters on that
%         plate) but in the real plate typeface rather than a chart typeface,
%         so it matches better when it covers the reading. Requires
%         BanglaRefLabels.
out = struct('Available', false, 'Reason', '');

useRef = ~isempty(cfg.BanglaRefImage);

if useRef && isempty(cfg.BanglaRefLabels)
    out.Reason = ['BanglaRefImage was given but BanglaRefLabels is empty. ', ...
                  'Supply one label per character, left to right, or clear ', ...
                  'BanglaRefImage to use the built-in chart alphabet instead.'];
    warning('main:noBanglaLabels', '%s', out.Reason);
    return;
end

refPath = '';   %#ok<NASGU> only read when useRef, but keeps the scope obvious
if useRef
    refPath = cfg.BanglaRefImage;
    if ~exist(refPath, 'file')
        refPath = fullfile(fileparts(mfilename('fullpath')), cfg.BanglaRefImage);
    end
    if ~exist(refPath, 'file')
        out.Reason = sprintf('Bangla reference image not found: %s', ...
                             cfg.BanglaRefImage);
        warning('main:banglaRefMissing', '%s', out.Reason);
        return;
    end
end

try
    if useRef
        ref = loadAndPreprocess(refPath, 'MaxWidth', cfg.MaxWidth, ...
                                'Denoise', false);
        tpl = banglaTemplates('Source', 'image', 'Image', ref, ...
            'Labels', cfg.BanglaRefLabels, 'Visualize', cfg.ShowDiagnostics);
        vprintf(cfg, '        templates: %d glyph(s) cut from %s\n', ...
            numel(tpl.Images), cfg.BanglaRefImage);
    else
        tpl = banglaTemplates('Source', cfg.BanglaTemplateSource, ...
            'Include', cfg.BanglaInclude, 'Visualize', cfg.ShowDiagnostics);
        vprintf(cfg, '        templates: %d class(es) from the ''%s'' charts (%s)\n', ...
            numel(tpl.Images), cfg.BanglaInclude, cfg.BanglaTemplateSource);
    end

    res = recognizeBanglaPlate(enhanced, tpl, ...
        'GroundTruth', cfg.BanglaGroundTruth, ...
        'Visualize', cfg.ShowFigures);

    out           = res;
    out.Available = true;
    out.Reason    = '';
    out.Templates = tpl;

    vprintf(cfg, '        recognised "%s"\n', res.Text);
    if isfield(res, 'TextUnicode') && ~isempty(res.TextUnicode)
        % Bengali only prints legibly on a console with a Bengali font. The
        % ASCII line above is the authoritative one; this is a convenience.
        vprintf(cfg, '        in Bengali "%s"\n', res.TextUnicode);
    end
    vprintf(cfg, '        %d character(s), mean NCC %.3f\n', ...
        res.NumChars, res.Confidence);
    if isfield(res, 'Accuracy')
        vprintf(cfg, '        accuracy %d/%d (%.0f%%)\n', ...
            res.Correct, numel(cfg.BanglaGroundTruth), 100 * res.Accuracy);
    end
catch ME
    out.Reason = ME.message;
    warning('main:banglaFailed', 'Bangla recognition failed: %s', ME.message);
end
end

% =======================================================================
function outDir = saveOutputs(outDir, r, cfg)
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Saved with plain clipping, matching what restoreImage returns -- no extra
% min-to-max stretch, so the PNGs look exactly like the figures.
imwrite(im2uint8(clip01(r.ROI)),            fullfile(outDir, 'roi_blurred.png'));
imwrite(im2uint8(clip01(r.RestoredWiener)), fullfile(outDir, 'restored_wiener.png'));
imwrite(im2uint8(clip01(r.RestoredLucy)),   fullfile(outDir, 'restored_lucy.png'));
imwrite(im2uint8(clip01(r.RestoredTV)),     fullfile(outDir, 'restored_tv.png'));
if ~isempty(r.RestoredDeepRGB)
    imwrite(im2uint8(clip01(r.RestoredDeepRGB)), fullfile(outDir, 'restored_deep.png'));
    imwrite(im2uint8(clip01(r.RestoredDeep)),    fullfile(outDir, 'restored_deep_gray_roi_size.png'));
end
if ~isempty(r.Reconstructed)
    imwrite(im2uint8(clip01(r.Reconstructed)), fullfile(outDir, 'reconstructed_plate.png'));
end
imwrite(im2uint8(clip01(r.EnhancedWiener)), fullfile(outDir, 'enhanced_wiener.png'));
imwrite(im2uint8(clip01(r.EnhancedLucy)),   fullfile(outDir, 'enhanced_lucy.png'));
imwrite(r.BinaryWiener,                     fullfile(outDir, 'binary_wiener.png'));
imwrite(r.BinaryLucy,                       fullfile(outDir, 'binary_lucy.png'));
% The chosen primary result -- this is "the deblurred plate" the pipeline
% stands behind, restored with cfg.RestoreMethod (TV by default).
imwrite(im2uint8(clip01(r.RestoredMain)),   fullfile(outDir, 'restored_main.png'));
imwrite(im2uint8(clip01(r.EnhancedMain)),   fullfile(outDir, 'enhanced_main.png'));
imwrite(r.BinaryMain,                       fullfile(outDir, 'binary_main.png'));
% The PSF is a kernel, not a photo, so min-to-max IS the right scaling here.
imwrite(im2uint8(mat2gray(r.PSF)),          fullfile(outDir, 'psf.png'));

% Figures, if any were drawn.
if isfield(r, 'Figures')
    fn = fieldnames(r.Figures);
    for k = 1:numel(fn)
        f = r.Figures.(fn{k});
        if ~isgraphics(f), continue; end
        target = fullfile(outDir, sprintf('figure_%s.png', lower(fn{k})));
        try
            exportgraphics(f, target, 'Resolution', 150);
        catch
            try
                print(f, target, '-dpng', '-r150');
            catch
                % not fatal
            end
        end
    end
end

writeSummary(fullfile(outDir, 'summary.txt'), r, cfg);
end

% =======================================================================
function writeSummary(path, r, cfg)
% UTF-8 explicitly, because the Bangla block below writes real Bengali. The
% default encoding on a Windows install is a legacy codepage that cannot
% represent it, and fprintf would silently substitute question marks.
fid = fopen(path, 'w', 'n', 'UTF-8');
if fid < 0
    return;
end
fprintf(fid, 'License plate deblurring summary\n');
fprintf(fid, 'generated %s\n\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));

if cfg.UseSyntheticTest
    fprintf(fid, 'source            : synthetic ("%s")\n', r.Truth.Text);
    fprintf(fid, 'true blur         : %g px at %g deg\n', r.Truth.Length, r.Truth.Angle);
else
    fprintf(fid, 'source            : %s\n', cfg.InputImagePath);
end
fprintf(fid, 'text polarity     : %s\n', r.Polarity);
fprintf(fid, 'ROI               : %dx%d (%s)\n', ...
    size(r.ROI, 2), size(r.ROI, 1), r.ROIInfo.Method);
fprintf(fid, 'estimated blur    : %g px at %g deg\n', r.Blur.Length, r.Blur.Angle);
if isfield(r.Blur.Info, 'Confidence') && ~isnan(r.Blur.Info.Confidence)
    fprintf(fid, 'estimate conf.    : %.2f\n', r.Blur.Info.Confidence);
end
if isfield(r, 'Truth')
    fprintf(fid, 'estimation error  : %+.1f px, %+.1f deg\n', ...
        r.Truth.LengthError, r.Truth.AngleError);
end
fprintf(fid, 'NSR / Lucy iters  : %.3g / %d\n', cfg.NSR, cfg.LucyIter);
fprintf(fid, 'TV weight / iters : %.4g / %d\n', cfg.TVWeight, cfg.TVIters);
fprintf(fid, 'primary restore   : %s\n', r.RestoreMethodUsed);
if ~isempty(r.PlateReading)
    rd = r.PlateReading;
    fprintf(fid, 'plate reading     : %s / %s  (%s)\n', rd.TopLine, rd.BottomLine, rd.Format);
    fn = fieldnames(rd.Confidence);
    fprintf(fid, 'reading confidence: %s\n', strjoin(cellfun(@(f) sprintf('%s %.2f', f, rd.Confidence.(f)), ...
        fn, 'UniformOutput', false).', ', '));
end
if r.Metrics.HasDeep
    fprintf(fid, 'deep learning     : sharpness gain %.2fx, %.2f s\n', ...
        r.Metrics.SharpnessGainDeep, r.DeepInfo.Seconds);
    if r.Metrics.HasReference
        fprintf(fid, 'PSNR deep / TV    : %.2f / %.2f dB\n', r.Metrics.PSNRDeep, r.Metrics.PSNRTV);
    end
end
fprintf(fid, 'sharpness gain    : Wiener %.2fx, Lucy %.2fx\n', ...
    r.Metrics.SharpnessGainWiener, r.Metrics.SharpnessGainLucy);
if r.Metrics.HasReference
    fprintf(fid, 'PSNR vs reference : blurred %.2f, Wiener %.2f, Lucy %.2f dB\n', ...
        r.Metrics.PSNRBlurred, r.Metrics.PSNRWiener, r.Metrics.PSNRLucy);
end
fprintf(fid, 'ringing (clipped) : Wiener %.1f%%, Lucy %.1f%%, TV %.1f%%\n', ...
    100 * r.RestoreInfo.WienerOvershoot, 100 * r.RestoreInfo.LucyOvershoot, ...
    100 * r.RestoreInfo.TVOvershoot);
if isfield(r, 'OCRWiener')
    fprintf(fid, 'OCR Wiener        : "%s" (conf %.2f, via %s)\n', ...
        r.OCRWiener.RecognizedText, r.OCRWiener.MeanConfidence, r.OCRWiener.Variant);
    fprintf(fid, 'OCR Lucy          : "%s" (conf %.2f, via %s)\n', ...
        r.OCRLucy.RecognizedText, r.OCRLucy.MeanConfidence, r.OCRLucy.Variant);
    if isfield(r.OCRWiener, 'CharacterAccuracy')
        fprintf(fid, 'OCR accuracy      : Wiener %.0f%%, Lucy %.0f%% (truth "%s")\n', ...
            100 * r.OCRWiener.CharacterAccuracy, ...
            100 * r.OCRLucy.CharacterAccuracy, r.OCRWiener.GroundTruth);
    end
end
if isfield(r, 'Bangla')
    if r.Bangla.Available
        fprintf(fid, 'Bangla text       : "%s"\n', r.Bangla.Text);
        if isfield(r.Bangla, 'TextUnicode') && ~isempty(r.Bangla.TextUnicode)
            fprintf(fid, 'Bangla (Bengali)  : %s\n', r.Bangla.TextUnicode);
        end
        if isfield(r.Bangla, 'Templates')
            fprintf(fid, 'templates         : %d class(es), source ''%s''\n', ...
                numel(r.Bangla.Templates.Images), r.Bangla.Templates.Source);
        end
        fprintf(fid, 'Bangla chars      : %d found, mean NCC %.3f\n', ...
            r.Bangla.NumChars, r.Bangla.Confidence);
        fprintf(fid, 'per-char NCC      : %s\n', ...
            strjoin(arrayfun(@(v) sprintf('%.2f', v), r.Bangla.Scores, ...
                             'UniformOutput', false), ' '));
        if isfield(r.Bangla, 'Accuracy')
            fprintf(fid, 'Bangla accuracy   : %d/%d (%.0f%%)\n', ...
                r.Bangla.Correct, numel(cfg.BanglaGroundTruth), ...
                100 * r.Bangla.Accuracy);
        end
        % A low character count is nearly always segmentation, not matching:
        % blur merges glyphs under the headline. Say so in the file, because
        % this is the failure a reader will otherwise misdiagnose.
        if ~isempty(cfg.BanglaGroundTruth) && ...
                r.Bangla.NumChars ~= numel(cfg.BanglaGroundTruth)
            fprintf(fid, '  note            : character count is wrong, so this is a\n');
            fprintf(fid, '                    SEGMENTATION failure -- the restoration was\n');
            fprintf(fid, '                    not sharp enough to reopen the gaps between\n');
            fprintf(fid, '                    glyphs. Matching never got a fair chance.\n');
        end
    else
        fprintf(fid, 'Bangla text       : not attempted (%s)\n', r.Bangla.Reason);
    end
end
fclose(fid);
end

% =======================================================================
% Small helpers
% =======================================================================
function vprintf(cfg, varargin)
if cfg.Verbose
    fprintf(varargin{:});
end
end

function p = resolvePath(p, thisDir)
if ~isempty(p) && exist(p, 'file') ~= 2
    candidate = fullfile(thisDir, p);
    if exist(candidate, 'file') == 2
        p = candidate;
    end
end
end

function e = wrapAngleError(d)
e = mod(d + 90, 180) - 90;
end

function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end
