# IDIF Pipeline

Image-derived input function (IDIF) from the carotid arteries in dynamic PET, using a carotid segmentation from the MRI. In this study's case, the MRI was a TOF-MRA, but the pipeline should work with any MRI that has already segmentations of the carotids. 

| Part | What it does |
| --- | --- |
| **Initial registration** | Rigid MRI-to-PET registration; carotid mask moved to PET space; first TACs |
| **Carotid refinement** | Four rigid registration strategies to the summed early PET (0-3 min), compared against a resampling-only baseline |
| **Final TACs** | Picks the strategy with the highest carotid peak, applies PVC, writes the final TACs |

## Before you run

<details>
<summary><b>Motion correction (MOCO)</b> – run beforehand</summary>

<br>

Not included in this pipeline. Run [FALCON](https://github.com/ENHANCE-PET/FALCON) first and place the output volumes as described under *Input data*. They are only needed for the final TACs.

> Sundar LKS, Lassen ML, Gutschmayer S, Ferrara D, Calabrò A, Yu J, Kluge K, Wang Y, Nardo L, Hasbak P, Kjaer A, Abdelhafez YG, Wang G, Cherry SR, Spencer BA, Badawi RD, Beyer T, Muzik O. Fully Automated, Fast Motion Correction of Dynamic Whole-Body and Total-Body PET/CT Imaging Studies. J Nucl Med. 2023. doi:10.2967/jnumed.122.265362

</details>

<details>
<summary><b>Partial-volume correction (PVC)</b> – included, PETPVC must be installed</summary>

<br>

Performed with [PETPVC](https://github.com/UCL/PETPVC) and must be installed separately (check with `petpvc --help`). Put it on your `PATH`, or set `PVC_CONDA_ENV` in the config. Set `PVC_FWHM` to your scanner resolution in mm.


> Thomas BA, Cuplov V, Bousse A, Mendes A, Thielemans K, Hutton BF, Erlandsson K. PETPVC: a toolbox for performing partial volume correction techniques in positron emission tomography. Phys Med Biol. 2016;61(22):7975-7993. doi:10.1088/0031-9155/61/22/7975

</details>


## Quick start

Create a Python 3.9+ environment containing `numpy`, `pandas`, `SimpleITK`, `matplotlib` and `seaborn`. They are also listed in [requirements.txt](requirements.txt).

1. Organise your data as described under *Input data*.
2. Edit [idif_config.py](idif_config.py) to match your own data (paths, subject IDs, scanner resolution). Each setting is commented.
3. Run:

```bash
python idif_pipeline.py
```



## Input data

<details>
<summary><b>Folder layout</b></summary>

```
DATA_DIR/
  subject_001/
    subject_001_PET_recon.nii.gz     4D dynamic PET
    subject_001_MRI.nii.gz           3D MRI
    Segmentation_cut.nii.gz          carotid segmentation (labels > 0)
    <MOCO_DIR_PREFIX>*/Motion-corrected-images/
      no_moco_vol_0000.nii.gz ...    frames before N_FRAMES
      moco_vol_0018.nii.gz ...       frames from N_FRAMES onward
```

</details>

## Usage

<details>
<summary><b>Run a single part or subject</b></summary>

```python
from idif_pipeline import run_initial_registration, run_carotid_refinement, run_final_tacs

run_initial_registration("subject_001")
run_carotid_refinement("subject_001")
run_final_tacs("subject_001")
```

Each part depends on the previous one, so run them in order.

</details>

## Output

<details>
<summary><b>Files written to <code>OUTPUT_DIR/&lt;subject_id&gt;/</code></b></summary>

| File | Content |
| --- | --- |
| `*_registered_mri.nii.gz`, `*_registered_mask.nii.gz` | MRI and mask in PET space |
| `*_tac_comparison.csv/.png` | TACs with and without registration |
| `*_registration_displacement.csv` | Rigid transform shift and rotation |
| `refined_segmentations/` | Masks, transforms and TACs of all candidate methods |
| `pvc_corrected_winner_FWHM_<x>/` | PVC-corrected frames |
| `*_tac_refined_winner.csv` | Final TACs: `raw`, `moco`, `moco_pvc` |
| `winner_peak_summary.csv` | Peak value of each candidate method |

</details>

## Method notes

<details>

<br>

- Registrations run single-threaded with a fixed random seed.
- The superior-inferior axis is detected from the image direction; a warning is printed if the segmentation looks mis-oriented.
- The carotid is split into a "petrous" segment (± `BUFFER_SLICES` around the peak slice) and the remaining "rest".

</details>
