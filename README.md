# KIDNEY-GIRLS
Authors: 
- 


Kidney Project
This project loads TLSC single-frame .nc images of rat kidneys, normalizes and denoises them, enhances vessel-like structures, and produces binary vessel masks with light post-processing. It’s designed to run robustly across a small batch of similar images with slightly different intensity ranges.

APM_KidneyProject/
  main.py                 # pipeline entry point (run this)
  readAndOpenImage.py     # open datasets, quick multi-image viewer
  normalizeImage.py       # normalization helpers (percentile-based)
  smoothing.py            # TV/bilateral/NLM/B-spline/FFT
  postProcessing.py       # vesselness + adaptive threshold + morphology
  plotting.py             # small helper to draw panels/overlays
  README.md



Key scripts & functions

***readAndOpenImage.py***
- open_and_show_one_frame(path): load .nc, return (ds, da, px_um)
- view_images(folder): quick side-by-side viewer for all .nc in a folder

***normalizeImage.py***
- percent_norm(img, p_lo=1, p_hi=99): robust [0,1] normalization
- normAdjustedThre(da, ...): auto high-clip normalization (clip top fraction → stretch)

***smoothing.py / smoothSF***
- TV-Chambolle, bilateral, NLM, B-spline, FFT low-pass (optional plots)

***postProcessing.py***
- enhanceVessels(img): Sato vesselness (on inverted or black_ridges), scales to [0,1]
-chooseThreshold(img, target=..., lo=..., hi=..., samples=..., dark=False): adaptive threshold by fraction (works on vesselness or normalized image; dark=True for bottom-tail)
- morphologicalOps(mask, original_img, px_um, min_obj_area_um2): remove small objects by physical area; overlay visualization

***plotting.py***
- plotResults(figures): quick panel plotting (image + optional colorbar + overlay)

***main.py***
- Wires everything together: open → normalize → denoise → enhance → threshold → clean → show/save.