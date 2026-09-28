%RUNDEEPDEBLURDEMO Deblur, read and reconstruct every test photo in data/.
%   Runs deblurPlate on each blurred image, saves the individual results to
%   data/<name>_deblurred_DL/ and writes one overview figure,
%   data/deep_deblur_overview.png: blurred input, network restoration, and
%   the clean plate reconstructed from what the reader read (with the reading
%   printed underneath).
%
%   Set compareClassical = true to add the cepstral-PSF Wiener and TV
%   restorations as extra columns (slower).

compareClassical = false;

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'src'));

files = { ...
    'data/photo_1_2026-08-26_01-46-56.jpg'
    'data/photo_2_2026-08-26_01-46-56.jpg'
    'data/photo_3_2026-08-26_01-46-56.jpg'
    'data/blur.png'
    'data/dhaka.png'
    };
files = files(cellfun(@(f) exist(fullfile(here, f), 'file') == 2, files));

n = numel(files);
nc = 3 + 2 * compareClassical;
fig = figure('Name', 'PlateDeblurNet on the project test images', 'NumberTitle', 'off', ...
             'Color', 'w', 'Position', [60 60 300 * nc 170 * n]);
t = tiledlayout(n, nc, 'Padding', 'compact', 'TileSpacing', 'compact');
title(t, 'Deblur, read, reconstruct (PlateDeblurNet + PlateReader)', 'FontWeight', 'bold');

for k = 1:n
    r = deblurPlate(fullfile(here, files{k}), 'ShowFigure', false, 'Classical', compareClassical);
    [~, nm] = fileparts(files{k});
    nexttile; imshow(r.ROI); title(strrep(nm, '_', '\_'), 'FontSize', 8);
    if compareClassical
        nexttile; imshow(min(max(r.Wiener, 0), 1)); title('Wiener', 'FontSize', 8);
        nexttile; imshow(min(max(r.TV, 0), 1));     title('TV', 'FontSize', 8);
    end
    nexttile; imshow(r.Restored); title('PlateDeblurNet', 'FontSize', 8);
    nexttile;
    if isfield(r, 'Reconstructed')
        imshow(r.Reconstructed);
        title(sprintf('%s  %s  (conf %.2f)', r.Reading.TopLine, r.Reading.BottomLine, ...
              r.Reading.MinConfidence), 'FontSize', 8);
    else
        axis off; title('reader model not found', 'FontSize', 8);
    end
end

out = fullfile(here, 'data', 'deep_deblur_overview.png');
try
    exportgraphics(fig, out, 'Resolution', 150);
catch
    print(fig, out, '-dpng', '-r150');
end
fprintf('Overview written to %s\n', out);
