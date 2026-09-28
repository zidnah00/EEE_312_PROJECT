function ok = checkSetup()
%CHECKSETUP Verify that everything the deep-learning path needs is present.
%   checkSetup
%
%   Checks the model files, the Image Processing Toolbox and MATLAB version,
%   then runs the whole pipeline on one test photo and prints what it read.
%   Run this first if anything behaves oddly.

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'src'));
ok = true;

fprintf('MATLAB %s\n', version('-release'));
if verLessThan('matlab', '9.1')
    fprintf(2, 'FAIL  MATLAB R2016b or newer is required (string functions, implicit expansion).\n'); ok = false;
else
    fprintf('OK    MATLAB version\n');
end

if exist('imresize', 'file') && exist('imbinarize', 'file')
    fprintf('OK    Image Processing Toolbox\n');
else
    fprintf(2, 'FAIL  Image Processing Toolbox is required.\n'); ok = false;
end

files = {'models/plateDeblurNet.mat', 'models/plateReader.mat', 'models/plateGlyphs.mat', ...
         'src/deblurPlateDL.m', 'src/recognizePlateDL.m', 'src/reconstructPlate.m'};
for k = 1:numel(files)
    f = fullfile(here, files{k});
    if exist(f, 'file') == 2
        d = dir(f);
        fprintf('OK    %-34s %6.1f MB\n', files{k}, d.bytes / 1e6);
    else
        fprintf(2, 'FAIL  %s is missing\n', files{k}); ok = false;
    end
end

if ~ok
    fprintf(2, '\nSomething is missing -- see above.\n');
    return;
end

img = fullfile(here, 'data', 'photo_1_2026-08-26_01-46-56.jpg');
if exist(img, 'file') ~= 2
    fprintf('\nNo test photo found, skipping the run test.\n');
    return;
end
fprintf('\nRunning the full pipeline on %s ...\n', 'data/photo_1_2026-08-26_01-46-56.jpg');
t = tic;
r = deblurPlate(img, 'ShowFigure', false, 'SaveDir', 'none');
fprintf('Read "%s / %s" in %.1f s (expected "চট্ট মেট্রো-গ / ১২-৪৭৩৩")\n', ...
    r.Reading.TopLine, r.Reading.BottomLine, toc(t));
fprintf('Lowest field confidence %.2f\n', r.Reading.MinConfidence);
fprintf('\nSetup looks good. Try:  r = deblurPlate(''data/photo_2_2026-08-26_01-46-56.jpg'');\n');
end
