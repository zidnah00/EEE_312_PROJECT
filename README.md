# License Plate Motion Deblurring (MATLAB)

Recovers readable text from blurred Bangla license plate photographs, then
reads the plate.

There are two restorers:

* **Read and rebuild (the strongest result):** PlateReader recognises the
  city, class letter and six digits straight from the blurred photo, and the
  plate is typeset clean and sharp from that reading. See
  [Reading the plate](#reading-the-plate-and-rebuilding-it-platereader).
* **Deep-learning restoration:** PlateDeblurNet, a convolutional neural
  network trained for Bangla plates. It learns the whole blurred → sharp
  mapping, so it needs no blur estimate. See
  [Deep-learning deblurring](#deep-learning-deblurring-platedeblurnet).
* **Classical (kept as the baseline):** the blur is modelled as linear motion,
  estimated (length and angle), and inverted with Wiener, Lucy-Richardson or TV
  deconvolution.

## Check your setup first

```matlab
checkSetup        % verifies toolboxes and model files, then reads one test photo
```

## Reading the plate and rebuilding it (PlateReader)

Pixel restoration alone could not make heavy blur legible, so the pipeline now
does what a person does: **it reads the plate and writes it out again.**

```matlab
r = deblurPlate('data/photo_1_2026-08-26_01-46-56.jpg');   % restore + read + reconstruct
r.Text                      % 'চট্ট মেট্রো-গ ১২-৪৭৩৩'
r.Reading.Confidence        % probability per field
imshow(r.Reconstructed)     % the clean plate, typeset from the reading
```

### Why this works where deblurring alone fails

A Bangladeshi plate has a fixed layout and a closed alphabet, so reading it is
nine small classification problems: layout (3), city or district (66), class
letter (29) and six digits (10 each). Choosing between 10 digits needs far less
information than redrawing the digit's pixels, so it still works on a smear
that no deconvolution can invert.

`src/recognizePlateDL.m` (PlateReader) reads the encoder features of
PlateDeblurNet on a 12x24 grid. For each field a small branch predicts a
spatial **attention** map -- "where is digit 3 on this plate?" -- and the
attended feature vector goes to a classifier. All six digits share one
classifier, so every digit of every training plate trains the same weights.
Training also supervises *where* to look, since the generator knows where it
drew each character. Without that supervision the class-letter attention got
stuck on the plate's top edge and never learned.

`src/reconstructPlate.m` then typesets the reading from a glyph atlas
(`models/plateGlyphs.mat`, Kalpurush, pre-shaped conjuncts) as a clean plate,
taking the plate and text colours from the photo. **Characters below 50%
confidence are drawn in red**, so an uncertain guess never looks like a fact.

### Accuracy

Unseen synthetic plates (384 plates, the same generator as training but a seed
never used for it), by blur severity:

| blur | digit accuracy | all 6 digits right | city | class letter | whole plate right |
|---|---|---|---|---|---|
| mild | 99.7% | 98% | 95% | 71% | 66% |
| medium | 97% | 87% | 91% | 59% | 52% |
| heavy | 95% | 81% | 88% | 48% | 41% |
| all | 97% | 89% | 91% | 60% | 53% |

Your three real blurred photos, whose sharp versions were kept out of training:

| photo | read | true | |
|---|---|---|---|
| photo_1 | চট্ট মেট্রো-গ ১২-৪৭৩৩ | চট্ট মেট্রো-গ ১২-৪৭৩৩ | all correct |
| photo_2 | চট্ট মেট্রো-গ ১১-১৬৩৮ | চট্ট মেট্রো-গ ১১-৬৫৩৮ | see figure |
| photo_3 | চট্ট মেট্রো-গ ১২-১১৭৮ | চট্ট মেট্রো-গ ১২-১১৭৮ | all correct |

`data/deep_learning_results/real_test_reconstruction.png` shows, for each:
blurred input, PlateDeblurNet restoration, reconstruction, and the true plate.
`user_blurred_reconstruction.png` does the same for the blurred photos you
made from your own clear plates (marked "seen plate" where that plate's sharp
version is in the training data, so only the "UNSEEN" row is a fair test).

### What to trust

* **Digits are the reliable part** (about 97% correct each, 100% at mild blur).
* **The class letter is the weak part** (60%). It is one small glyph, one
  per plate, out of 29 -- the least-trained classifier in the model. Check it
  against the input, and treat a red character as "unknown", not as a reading.
* The reconstruction is always sharp, because it is typeset. Sharpness is
  therefore **not** evidence that the reading is right -- the confidence
  numbers are.

## Deep-learning deblurring (PlateDeblurNet)

```matlab
r = deblurPlate('data/photo_1_2026-08-26_01-46-56.jpg');      % one plate crop -> figure + saved PNGs
r = deblurPlate('car.jpg', 'ROI', 'manual');                   % whole-car photo: drag a box round the plate
r = deblurPlate('data/photo_2_2026-08-26_01-46-56.jpg', 'Classical', true);  % side by side with Wiener / TV
runDeepDeblurDemo                                               % every test photo -> data/deep_deblur_overview.png
out = deblurPlateDL(imread('plate.jpg'));                       % just the network, returns RGB in [0,1]
r = main(struct('InputImagePath','data/photo_1_2026-08-26_01-46-56.jpg','Script','bangla'));  % full pipeline
```

### Why the classical pipeline failed on real photos

Wiener, Lucy-Richardson and TV only work when the photo really is
*sharp plate ⊛ one straight-line PSF*, and when that PSF is estimated almost
exactly. The README sections below measure how fragile that is: ±4 px of
length error already costs accuracy. The real test photos break the model in
several ways at once:

* the smear is real shake rather than a clean line
* the images are low resolution, JPEG-compressed and washed out
* Bangla glyphs are thin and close together, so inversion ringing destroys them

### What the network is

`src/deblurPlateDL.m` + `models/plateDeblurNet.mat`:

* **Architecture:** a U-Net with residual blocks (6.7 M parameters). It
  downsamples to 1/16 resolution, so its receptive field covers the whole plate
  and it can undo long smears. It also upsamples the result 2×.
* **Input and output:** the plate crop is resized to 96 px high and
  auto-contrasted. The output is 192 px high.
* **No toolbox needed:** the forward pass is written in plain MATLAB (matrix
  products, reshapes, `max`), so it runs without the Deep Learning Toolbox.

**Training** (full details in `deep_learning/README.md`):

* **Data:** ~70 000 image pairs, generated on the fly.
  * Sharp targets are two-line Bangladeshi plates rendered in Kalpurush,
    SolaimanLipi, Nirmala UI, Vrinda and Free fonts, plus real plates cut from
    `data/Clear_Plates`.
  * The blurred inputs add random camera-shake trajectories, linear motion at
    any angle (mostly horizontal), ghosting, defocus, 36–400 px capture
    resolution, noise, JPEG at quality 20–95, haze and colour casts.
* **Held out:** the three plates in `Clear_Plates/photo_3_2026-09-11_12-49-01.jpg`
  are the sharp versions of the real blurred test photos
  `data/photo_{1,2,3}_2026-08-26_01-46-56.jpg`. They were **excluded from
  training**, so the network has never seen the plates it is tested on.

### Results

**Real test photos.** File: `data/deep_learning_results/real_photos_comparison_labeled.png`.
Columns: blurred input | previous result of this project (Wiener/TV) |
PlateDeblurNet | the true sharp plate.

The number line comes back readable on all three photos (১২-৪৭৩৩, ১১-৬৫৩৮,
১২-১১৭৮), where the deconvolution output isn't. The upper line (চট্ট মেট্রো-গ)
is recovered only roughly. Its glyphs are smaller than the length of the
smear, so much of their detail is simply gone from the photo.

**Synthetic benchmark.** 60 plates generated with a seed that was never used in
training, covering mild to very heavy blur. The Wiener row is given the
**true** PSF and the best NSR per image. That is an upper bound the real
pipeline never reaches, because it has to estimate the PSF.

| blur level (20 plates each) | blurred input (PSNR / SSIM) | Wiener, true PSF | PlateDeblurNet |
|---|---|---|---|
| mild | 18.43 dB / 0.683 | 19.21 dB / 0.584 | **20.47 dB / 0.799** |
| medium | 15.31 dB / 0.552 | 16.45 dB / 0.447 | **17.28 dB / 0.711** |
| heavy | 14.06 dB / 0.508 | 14.37 dB / 0.428 | **15.93 dB / 0.655** |
| all 60 | 15.93 dB / 0.581 | 16.67 dB / 0.486 | **17.89 dB / 0.722** |

The network beats oracle-PSF Wiener at every blur level, by about 1.2 dB PSNR
and 0.24 SSIM on average. Wiener's SSIM is even *below* that of the blurred
input, because its ringing destroys structure.

### Honest limits

* **Crop first.** The network expects a crop around one plate. A little margin
  or a small tilt is fine (it was trained with both). For a whole-car photo,
  use `'ROI','manual'`.
* **Some blur can't be undone.** When the smear is much longer than the
  character height, the information is physically gone. The network then
  produces the most plausible Bangla plate, which can contain wrong small
  glyphs. Trust the digits more than the small upper-line text, and check
  against the input.
* **Latin plates are outside the training data.** It was trained on Bangla
  plates. On Latin plates (e.g. `img4.png`) it still sharpens, but it's
  outside what it learned.
* **Retraining is possible.** Everything to retrain or extend it (more fonts,
  more real plates) is in `deep_learning/`.

## Quick start (classical pipeline and main)

```matlab
r = main();                                        % real photo, data/blur.png
r = main(struct('UseSyntheticTest', true));        % self-scoring demo
r = main(struct('InputImagePath', 'data/dhaka.png'));
r = main(struct('InputImagePath', 'data/img2.png', 'ROIMethod', 'auto'));
r = main(struct('InputImagePath', 'data/img4.png', 'ROIMethod', 'manual'));
```

For Bengali plates see [Bangla plates](#bangla-plates) below — English OCR
cannot read them at all, so `Script`, `'bangla'` swaps in a different stage 9,
which needs no configuration beyond that one flag:

```matlab
r = main(struct('InputImagePath', 'data/blur.png', 'Script', 'bangla'));
```

`main` is a function, not a script. It takes an optional config struct where
every field is optional, and returns one results struct holding the images, the
estimated blur parameters, the quality metrics and the OCR output, so it can be
scripted or swept without reading anything off a figure. Outputs are also
written to `data/<name>_deblurred/` as PNGs plus a `summary.txt`.

## Pipeline

| Stage | Function | What it does |
|---|---|---|
| 1 | `loadAndPreprocess` | read, grayscale, optional downscale, mild median denoise, detect text polarity |
| 2 | `extractROI` | crop the plate (`full`, `auto` or `manual`) |
| 3 | `analyzeFFTSpectrum` | Radon/projection-variance angle, kept as a **cross-check** |
| 4 | `estimateBlurParameters` | cepstral length + angle estimate (the real estimator) |
| 5 | `generatePSF` | `fspecial('motion', len, angle)`, energy-normalised |
| 6 | `applyEdgeTaper` | suppress boundary ringing before deconvolution |
| 7 | `restoreImage` | Wiener, damped Richardson–Lucy and TV (classical baseline) |
| 7b | `deblurPlateDL` | **PlateDeblurNet deep-learning restoration — the default result** (`RestoreMethod`, `'deep'`) |
| 8 | `enhanceAndBinarize` | CLAHE, bottom-hat text isolation, adaptive threshold |
| 9 | `ocrAndEvaluate` | OCR across several presentations, scored against ground truth |
| 9b | `segmentBanglaPlate` → `banglaTemplates` → `recognizeBanglaPlate` | the Bengali route, used instead of stage 9 when `Script` is `'bangla'` |

Helpers: `banglaCharset`, `estimateBlurCepstral`, `refineBlurParameters`,
`detectTextPolarity`, `makeSyntheticPlate`, `simulateMotionBlur`,
`sweepParameters`, `levenshteinDistance`.

## How the blur is estimated

The estimator is cepstral. Take the log-magnitude spectrum of a Hann-windowed,
zero-padded ROI and inverse-transform it; a linear motion blur leaves a negative
trough at a radius equal to the blur length, along the direction of motion.
Angle comes from scoring rays outward from the cepstrum centre and taking the
direction whose trough is deepest; length is then read off that ray.

Two details matter. The 180° mirror ambiguity is resolved by measuring
directional roughness on the *input* image and keeping the smoother direction —
motion smears along its own axis, so the image is smoother in that direction.
And the Radon angle from stage 3 is only a cross-check: when the two disagree by
more than 15° you get a warning, but the cepstral angle wins, because it
measured far more reliably.

On nine synthetic trials spanning 9–31 px and −15° to 90°, the cepstral estimate
passes 9/9 with a mean absolute length error of **0.06 px**.

## Why refinement is off by default

`refineBlurParameters` searches a narrow band around the seed and keeps whatever
maximises the restoration's gradient energy. It is available but disabled,
because measurement says it does not help:

| estimator | synthetic pass | mean \|length error\| |
|---|---|---|
| cepstral seed alone | 9/9 | **0.06 px** |
| band 0.15, bias 2.0 | 9/9 | 1.00 px |
| band 0.15, no bias | 8/9 | 1.44 px |
| band 0.25, 3% margin | 9/9 | 1.00 px |

On real photographs it is worse than the table suggests. Deconvolution ringing
is itself gradient energy, so the score climbs with length and the search drifts
long: it pushed `data/blur.png` from 57 px to 62–72 px and `data/dhaka.png` from
29 px to 36–38 px, in both cases past the visually sharpest result. Enable it
(`cfg.Refine = true`) only when `results.Blur.Info.Confidence` is low.

More generally, no-reference sharpness metrics — gradient sparsity, Laplacian
variance, gradient kurtosis, Otsu between-class variance, gradient entropy —
were all tried as global objectives and none was reliable. That is why the
method is spectral rather than search-based.

## Rescaling: the bug that made good restorations look bad

Deconvolution overshoots. A correct restoration of a plate typically lands in
about `[-0.24, 1.19]` rather than `[0,1]`. This code used to bring that back with
`mat2gray`, i.e. min-to-max, which lets a handful of ringing pixels at the
extremes set the scale for the whole image — the real content got squeezed into
roughly `[0.17, 0.81]` and the result looked flat and grey. Measured against the
sharp original:

| blur | `mat2gray` | clipping |
|---|---|---|
| 21 px @ 12° | 14.77 dB | **17.65 dB** |
| 15 px @ 0° | 15.57 dB | **20.68 dB** |

Clipping also roughly doubled gradient energy in both cases, and on the real
light-on-dark plates the two agree to within 0.05 grey levels, so clipping is
never worse. `restoreImage` now clips by default (`'Rescale'` accepts `clip`,
`stretch` or `minmax`). For the same reason the figures use `imshow(x)` and never
`imshow(x, [])` — the empty-limits form would rescale each panel independently
and reintroduce exactly this problem at display time.

Note the distinction: min-to-max *is* correct for the input photograph and for
displaying the PSF kernel. It is only wrong on deconvolved output.

## Defaults, and why

| Parameter | Value | Reason |
|---|---|---|
| `NSR` | 0.02 | 0.005 sits deep in the ringing regime; 0.02–0.05 is the sweet spot, and adequate regularisation also buys tolerance to an imperfect PSF |
| `LucyIter` | 8, damped | Richardson–Lucy amplifies noise and PSF error much faster than Wiener; 20 undamped iterations speckled |
| `MaxLen` | 80 | the real test image needs 57 px; the old cap of 40 made a correct answer unreachable |
| `Refine` | false | see above |
| `ROIMethod` | `full` | the images in `data/` are already cropped plates. Use `manual` to draw the box yourself, `auto` for whole-car photos — but see below |
| `Rescale` | `clip` | see above |
| `RestoreMethod` | `deep` | PlateDeblurNet; falls back to `tv` if `models/plateDeblurNet.mat` is missing |
| `BanglaTemplateSource` | `charset` | builds the alphabet from the charts in `data/`, so `'Script','bangla'` works with no further setup |
| `BanglaInclude` | `plate` | all 60 chart classes minus the three combining marks, which never stand alone on a plate and collide with the digit one |

Edge tapering is not optional in practice — it was the main source of the
vertical striping in the original output.

## Choosing the ROI yourself

```matlab
r = main(struct('InputImagePath', 'data/img4.png', 'ROIMethod', 'manual'));
```

A window opens; drag a rectangle and double-click inside it to confirm. Drag —
a single click gives a zero-size box, which `extractROI` now catches and asks
again for, up to five attempts, rather than passing a 1-pixel ROI into stage 4.
A hand-drawn box is **not** padded (you already framed it); pass
`'PadManual', true` if you want the 12% margin back.

Prefer `manual`. Ported to Python and run over the real photographs in `data/`,
the `auto` search picked a region at the very top edge of `img4.png` nowhere
near the plate, returned only a fragment of the glyphs on `dhaka.png`, and found
nothing at all on `blur_image_1.jfif`. Edge density plus aspect-ratio filtering
is a weak detector on cluttered vehicle photos. The aspect band was widened from
2.0–6.0 to 1.1–7.0 so a two-line plate (nearer 1.3–2.5) cannot fall through the
floor, but be honest about what that changed: nothing, on this data. The
candidate *components* measure 2.07–4.03 and already passed the old band. It is
a precaution, not a fix.

## What was wrong originally

The symptom was that the final figure looked like noise. There was no single
cause; there were seven, and each was verified rather than assumed.

**Double angle correction.** `analyzeFFTSpectrum` returned
`mod(stripeAngle + 90, 180)` and `main.m` passed it `angleEst - 90`. The two
corrections cancelled, so the PSF was rotated 90° away from the true motion.
`analyzeFFTSpectrum` now applies exactly one correction and callers pass its
output through unmodified.

**An impossible length cap.** `main.m` simulated a 55 px blur and then searched
with `'MaxLen', 40`. The correct answer was outside the search space.

**Wrong text polarity.** The images in `data/` are light text on a dark
background, but `enhanceAndBinarize` hard-coded a dark-on-light bottom-hat, so
binarisation received the one polarity it could not handle. Polarity is now
detected (`detectTextPolarity`) and the image inverted when needed.

**A structuring element smaller than the characters.** The bottom-hat element was
a fixed 15×15, which hollows out any character taller than 15 px. It now scales
with ROI height.

**RGBA images passed through unconverted**, because the channel test was
`size(im,3) == 3`. Now handled by a `switch` on channel count.

**Wrong ground truth.** `groundTruthText` was `'hyft345'` — lowercase, while the
OCR character set is `A-Z0-9`. The synthetic demo also drew white text on a
`zeros()` canvas, i.e. the wrong polarity, so it could never produce a clean
result. `makeSyntheticPlate` now draws upper-case dark characters on a light
plate with a border, and clamps the simulated blur below the character height so
the glyphs are smeared rather than destroyed.

**Min-to-max rescaling of the deconvolution output**, described above.

An eighth, introduced later during this rework rather than inherited: the
parameter-sweep figure rendered one untitled panel and eleven empty ones.
`sweepParameters` computed its highlight tolerance as `max(diff(lenRange), 1)`,
and MATLAB's two-argument `max` is **elementwise, not a reduction** — for
lengths `[34 46 57 71]` it returned `[12 11 14]`, and feeding that 1×3 logical
into `&&` threw on the first tile. Now reduced to a scalar with
`min(abs(diff(v)))`. Worth remembering as a MATLAB trap: `max(v, 1)` clamps,
`max(v)` reduces.

## Bangla plates

MATLAB's `ocr` is Tesseract with an English model, restricted here to `A-Z0-9`.
It cannot read Bengali at all — not badly, *not at all*. So `Script`, `'bangla'`
replaces stage 9 with three functions that exploit a fact general OCR cannot: a
plate alphabet is **closed**. A district or metro name from a fixed list, one
class letter, and Bengali digits. Small closed-set classification by normalised
cross-correlation is a far easier problem, and it needs no Deep Learning Toolbox
— it is segmentation plus matrix arithmetic, by design.

```matlab
r = main(struct('InputImagePath','data/blur.png', 'Script','bangla'));
r.Bangla.Text          % ASCII labels, e.g. 'DHA AA KA AA'
r.Bangla.TextUnicode   % the same reading as real Bengali: ঢাকা
```

That is the whole configuration. The templates are built automatically from the
two glyph charts in `data/`, so there is nothing to label by hand. If you would
rather learn templates from one sharp reference plate — a real plate typeface is
not a chart typeface — pass it explicitly instead:

```matlab
r = main(struct( ...
    'InputImagePath',   'data/blur.png', ...
    'Script',           'bangla', ...
    'BanglaRefImage',   'data/dhaka_l.png', ...    % a SHARP reference
    'BanglaRefLabels',  {{'DHA','AA','KA','AA'}}, ...
    'BanglaGroundTruth',{{'DHA','AA','KA','AA'}}));
```

Note the **double braces**: `struct()` consumes one level of cell nesting.

### The headline is the whole problem

Bengali characters in a word are joined by the mātrā (shirorekha), the
horizontal bar along the top. Connected components therefore do not separate
them. Measured on `data/dhaka_l.png`:

| approach | components found |
|---|---|
| `bwconncomp` on the raw binary | **1** (useless) |
| after zeroing the headline rows | **4** — ঢ, া, ক, া |

The headline is easy to isolate because it is the only near-full-width row band:
rows 16–42 of that image covered 0.98–0.99 of the glyph width, while the
character bodies below covered 0.33–0.64. So `segmentBanglaPlate` finds that
band, zeros it, labels, then re-attaches the bar to each box before
normalisation. Bengali digits carry no headline, so on a two-line plate the
upper line has one and the lower line usually does not — both cases are handled.

### The alphabet, from the two charts in `data/`

`data/Bangla_Letters.jpg` and `data/bangla_numbers.jpg` are ordinary glyph charts,
and they are the template source. `banglaCharset` supplies the labels and Unicode
codepoints; `banglaTemplates('Source','charset')` cuts the glyphs and pairs the
two up, so the whole alphabet costs one call and no hand-labelling.

Both charts segment exactly, and the counts were checked against pixels rather
than assumed — each montage was rendered and read back glyph by glyph.
`Bangla_Letters.jpg` yields **50** glyphs in rows of 11/11/11/11/6, and that count
is unchanged for every column-gap tolerance from 1 to 12 px. `bangla_numbers.jpg`
yields **11**, not ten: it is a 5×2 flashcard grid whose last cell holds the
two-digit ১০, so the extractor returns ১–৯, then ১ again, then ০. That fits the
existing rule about repeated glyphs rather than needing a special case, and one
generic row-then-column extractor handles both charts.

Class separation across the resulting 60 classes is comfortable. No class has a
runner-up NCC above **0.82**, median runner-up is 0.52, and the single worst pair
is য vs য়, which differ only by a nukta dot — precisely the confusion you would
predict, which is itself a check that the measurement means something.

`'plate'` (the default `BanglaInclude`) is those 60 classes minus the three
combining marks, leaving **57**. The marks are not letters: they attach to a
preceding consonant and never stand alone on a plate, and normalising a two-dot
visarga into a 48×48 box makes it collide with the digit one — a real error the
full set commits and this set cannot.

### What deblurring actually buys you

This is the number that justifies stages 1–8. Chart glyphs scaled to a 64 px
plate-like height, motion blurred, Wiener restored, then matched:

| blur | raw blurred input | after restoration (57-class `'plate'` set) |
|---|---|---|
| 0 px | 60/60 | 57/57 — 100% |
| 9 px | — | 57/57 — 100% |
| 17 px | — | 55/57 — 96% |
| 27 px | — | 54/57 — 95% |
| 39 px | — | 51/57 — 89% |
| 55 px | **4/60** | 50/57 — 88% |

**Digits scored 10/10 at every single blur level tested.** That is the figure that
matters for a plate: the number line is what identifies the vehicle. Additive
noise up to σ = 0.04 barely moves any of this. Tolerance to a wrong PSF length is
roughly ±4 px — at −6 px accuracy falls to 46/60, at +6 px to 53/60.

The remaining failures at heavy blur are consistent rather than random: ন, য়, উ
and য. Those are shapes that differ from a neighbour by one small feature, so
they are the first things blur erases.

On whole words the same story holds. `data/dhaka_l.png` blurred synthetically,
matched against templates cut from the sharp original:

| blur | raw blurred input | after Wiener restoration |
|---|---|---|
| 15 px | 4/4 chars, NCC 0.971 | 4/4, NCC 0.994 |
| 27 px | 4/4 chars, NCC 0.814 | 4/4, NCC 0.961 |
| 39 px | **segmentation collapsed** | 4/4, NCC 0.938 |
| 45 px | **segmentation collapsed** | 4/4, NCC 0.941 |
| 55 px | **segmentation collapsed** | 4/4, NCC 0.912 |

Blur closes the gaps between glyphs, so past roughly 33 px the word merges into
one component and there is nothing left to classify. Restoration extends usable
Bengali reading from ~33 px of motion blur to at least 55 px.

The corollary matters when you debug: **always segment the restored image, never
the raw input.** If you get too few characters, suspect the deblurring before
you suspect the segmenter. On the mildly blurred `data/dhaka.png` the same code
returns 2 boxes instead of 4. `summary.txt` says so explicitly when the
character count disagrees with your ground truth.

### Templates: four routes, not equally trustworthy

| route | status | use it when |
|---|---|---|
| `'charset'` | **validated, the default** | you want the whole alphabet, no labelling |
| `'chart'` | validated | you have your own chart and your own labels |
| `'image'` | validated | you have a sharp reference plate in the real typeface |
| `'font'` | **not validated here** | superseded by `'charset'`; see below |

The font route renders glyphs with `insertText`, needs a Bengali-capable font
such as Nirmala UI or Vrinda, and because Bengali is a complex script requiring
text shaping, `insertText` is not guaranteed to compose dependent vowel signs and
conjuncts correctly. It was never confirmed working because no Bengali font was
available where this was built. `'charset'` makes it unnecessary.

One non-obvious implementation detail, recorded because it took an hour to find:
in the chart extractor, **small-component removal must happen after the grid
lines are erased, not before.** A speck of JPEG noise touching a grid line is
part of that line's component, so a global `bwareaopen` keeps it; it only becomes
an orphan once the line is gone. Despeckling too early left a stray pixel that
inflated one digit's bounding box to three times its proper width.

Repeat a label when a glyph occurs twice. The two া aa-kar signs in ঢাকা are one
class, not two; labelling them `AA-1` and `AA-2` makes accuracy meaningless,
because "distinguishing" identical shapes is a non-task.

### The honest caveat

The 57 classes come from a **chart typeface, not a plate typeface.** Bangladeshi
plates use their own letterforms, and matching a photographed plate against chart
glyphs will do worse than the numbers above, which were measured on chart glyphs
matched against chart glyphs. Nothing here is circular — blur, noise and PSF error
were all applied between template and test — but it is still the easier case.

`data/` accordingly still has **no labelled Bangladeshi plate.** `blur.png`,
`dhaka.png` and `dhaka_l.png` are the single word ঢাকা rendered white-on-black at
three blur levels (`dhaka_l` sharp, `dhaka` mild, `blur` heavy). `blur_image_1.jfif`
(WD-71817), `img3.jfif`, `img4.png` and `img5.png` are real cars carrying
single-line **Latin** plates, and `img5.png` is probably unrecoverable. A handful
of real two-line Bangladeshi plate photographs, with the reading written down,
remains the single most valuable thing to add: it converts the caveat above into
a measurement.

## Limitation worth knowing

Real motion blur is not exactly linear, so at long lengths aggressive inversion
amplifies model error rather than removing blur — this is what the `NSR` default
is protecting against. And the sweep grid's score is edge energy, which drifts
long; when the grid's orange (best score) and blue (estimator's pick) outlines
disagree, trust your eyes.

If you set a ground-truth *string* (`GroundTruth`), use it only for plates that
really do carry Latin characters; the synthetic demo is the end-to-end test for
the English path. For Bengali use `BanglaGroundTruth`, a cell array of labels.

## Requirements

MATLAB R2019b or newer (for `tiledlayout`, `startsWith`, implicit expansion) with the Image Processing Toolbox, which is genuinely required — `main`
now checks for it up front and errors with one clear sentence rather than
failing obscurely in stage 6. The Computer Vision Toolbox is needed only for
`ocr`, `insertText` and `insertShape`; without it the English stage 9 switches
itself off with a warning and stages 1–8 run normally. **No Deep Learning
Toolbox is needed anywhere**. The neural network in stage 7b is evaluated with
plain matrix products from `models/plateDeblurNet.mat`, and the Bengali reader
is segmentation plus normalised cross-correlation, so both run on any install.
Retraining the network (optional) needs Python — see `deep_learning/README.md`. No
Signal Processing Toolbox either; `hann` and `xcorr` were replaced with local
equivalents. `copy_of_main.m` is the author's debugging copy and is not part of
the pipeline.
