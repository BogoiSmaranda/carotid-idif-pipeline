import os

DATA_DIR = os.environ.get("IDIF_DATA_DIR", "/path/to/data")  # input root folder
OUTPUT_DIR = os.environ.get("IDIF_OUTPUT_DIR", os.path.join(DATA_DIR, "idif_output"))  # results folder
BASE_DIR = DATA_DIR  # alias for DATA_DIR

PET_FILENAME = "{id}_PET_recon.nii.gz"  # 4D PET filename
MRI_FILENAME = "{id}_MRI.nii.gz"  # 3D MRI filename
MASK_FILENAME = "Segmentation_cut.nii.gz"  # carotid segmentation filename

MOCO_DIR_PREFIX = "FALCONZ-V02-recon"  # recon folder prefix
MOCO_SUBDIR = "Motion-corrected-images"  # motion-corrected volumes folder
NO_MOCO_VOLUME_PATTERN = "no_moco_vol_{i:04d}.nii.gz"  # early frame volumes
MOCO_VOLUME_PATTERN = "moco_vol_{i:04d}.nii.gz"  # late frame volumes

PATIENT_IDS = [  # subject folder names
    "subject_001",
]

N_FRAMES = 18  # early frames (PET sum)
PET_REF_FRAME_INDEX = 17  # registration target frame
DEFAULT_LAST_FRAME = 34  # total frames used
LAST_FRAME_OVERRIDES = {}  # per-subject frame count

REGISTRATION_DEFAULTS = dict(  # initial registration
    multi_level=[4, 2, 1],  
    smoothing_sigmas=[2, 1, 0],  
    use_random=False,  
    sampling_percentage=0.05,  
    metric_bins=50,  
)

BUFFER_SLICES = 10  # slices around peak
SUMMARY_CSV_NAME = "all_patients_summary.csv"  # cohort summary file

N_WORKERS = 4  # parallel subjects

METHOD_KEYS = [  # candidate mask methods
    "identity",
    "initial_petrous_refined_full",
    "initial_petrous_refined_roi",
    "refine_rigid_reg_petrous",
    "refined_petrous_sum",
]

ROI_DILATION_RADIUS = 15  # ROI dilation (voxels)

REFINE_REGISTRATION_DEFAULTS = dict(  # refinement registration
    metric_bins=64,  
    sampling_percentage=0.20,  
    learning_rate=0.1,  
    num_iterations=300,  
    convergence_min_value=1e-7,  
    convergence_window_size=20,  
    shrink_factors=[2, 1],  
    smoothing_sigmas=[1.0, 0.0],  
)

PVC_FWHM = (6.4, 6.4, 6.4)  # scanner PSF (mm)
PVC_METHOD = "STC"  # PETPVC method
PVC_THREADS = 8  # parallel PVC jobs
PVC_DIR_NAME = f"pvc_corrected_winner_FWHM_{PVC_FWHM[0]}"  # PVC output folder
PVC_CONDA_ENV = os.environ.get("IDIF_PVC_CONDA_ENV", "")  # conda env (optional)
