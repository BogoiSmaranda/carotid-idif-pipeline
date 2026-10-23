import os
import csv
import subprocess

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import idif_config as cfg


def get_paths(patient_id):
    patient_dir = os.path.join(cfg.BASE_DIR, patient_id)
    pet_path = os.path.join(patient_dir, cfg.PET_FILENAME.format(id=patient_id))
    mri_path = os.path.join(patient_dir, cfg.MRI_FILENAME.format(id=patient_id))
    mask_path = os.path.join(patient_dir, cfg.MASK_FILENAME)
    return pet_path, mri_path, mask_path


# determine the longitudinal direction of the carotids
def get_superior_inferior_axis(mask_sitk: sitk.Image) -> int:
    direction = np.array(mask_sitk.GetDirection()).reshape(3, 3)
    si_vector = np.array([0, 0, 1])
    alignments = [abs(np.dot(direction[:, i], si_vector)) for i in range(3)]
    image_axis = int(np.argmax(alignments))
    return 2 - image_axis


def petrous_rest_masks(mask_3d, method_name, buffer_slices):
    arr = sitk.GetArrayFromImage(mask_3d)
    v_axis = get_superior_inferior_axis(mask_3d)
    print(f"  [{method_name}] Detected S-I axis: {v_axis} (0=Z, 1=Y, 2=X)")

    indices = np.where(arr > 0)
    if len(indices[0]) > 0:
        si_range = indices[v_axis].max() - indices[v_axis].min()
        longest = max(indices[i].max() - indices[i].min() for i in range(3))
        if si_range < 0.5 * longest:
            print(f"  [{method_name}] WARNING: S-I axis span ({si_range}) is much "
                  f"shorter than longest axis ({longest}). Check image orientation.")

    other_axes = tuple(i for i in range(3) if i != v_axis)
    summed_array = np.sum(arr, axis=other_axes)
    peak_idx = int(np.argmax(summed_array))
    z_indices = np.where(summed_array > 0)[0]
    z_start = max(int(z_indices.min()), peak_idx - buffer_slices)
    z_end = min(int(z_indices.max()), peak_idx + buffer_slices)

    petrous_arr = np.zeros_like(arr)
    slices = [slice(None)] * 3
    slices[v_axis] = slice(z_start, z_end + 1)
    petrous_arr[tuple(slices)] = arr[tuple(slices)]

    rest_arr = np.copy(arr)
    rest_arr[petrous_arr > 0] = 0

    petrous_mask = sitk.GetImageFromArray(petrous_arr)
    petrous_mask.CopyInformation(mask_3d)
    rest_mask = sitk.GetImageFromArray(rest_arr)
    rest_mask.CopyInformation(mask_3d)
    return petrous_mask, rest_mask


def extract_tac(pet_4d: sitk.Image, carotid_mask: sitk.Image) -> list:
    num_frames = pet_4d.GetSize()[3]
    tac = []
    stats_filter = sitk.LabelStatisticsImageFilter()
    for t in range(num_frames):
        frame_size = list(pet_4d.GetSize())
        frame_size[3] = 0
        frame_index = [0, 0, 0, t]
        pet_3d_frame = sitk.Extract(pet_4d, frame_size, frame_index)
        stats_filter.Execute(pet_3d_frame, carotid_mask)
        if stats_filter.HasLabel(1):
            tac.append(stats_filter.GetMean(1))
        else:
            tac.append(0.0)
    return tac


def extract_tac_frame(pet_3d, carotid_mask) -> float:
    stats_filter = sitk.LabelStatisticsImageFilter()
    stats_filter.Execute(pet_3d, carotid_mask)
    if stats_filter.HasLabel(1):
        return stats_filter.GetMean(1)
    return 0.0


def extract_tac_from_path(pet_4d_path, mask_3d) -> np.ndarray:
    pet_4d = sitk.ReadImage(pet_4d_path, sitk.sitkFloat32)
    n_frames = pet_4d.GetSize()[3]
    mask_uint8 = sitk.Cast(mask_3d > 0, sitk.sitkUInt8)
    tac = [extract_tac_frame(pet_4d[:, :, :, t], mask_uint8) for t in range(n_frames)]
    return np.array(tac)


def split_save_and_extract(pet_4d, mask_3d, method_name, output_dir, buffer_slices=10):
    petrous_mask, rest_mask = petrous_rest_masks(mask_3d, method_name, buffer_slices)

    sitk.WriteImage(petrous_mask, os.path.join(output_dir, f"{method_name}_petrous.nii.gz"))
    sitk.WriteImage(rest_mask, os.path.join(output_dir, f"{method_name}_rest.nii.gz"))
    sitk.WriteImage(mask_3d, os.path.join(output_dir, f"{method_name}_full_mask.nii.gz"))

    tac_petrous = extract_tac(pet_4d, petrous_mask)
    tac_rest = extract_tac(pet_4d, rest_mask)
    return tac_petrous, tac_rest


def split_petrous_and_extract(pet_4d_path, mask_3d, method_name, output_dir, buffer_slices=10):
    petrous_mask, rest_mask = petrous_rest_masks(mask_3d, method_name, buffer_slices)

    sitk.WriteImage(petrous_mask, os.path.join(output_dir, f"{method_name}_petrous.nii.gz"))

    tac_petrous = extract_tac_from_path(pet_4d_path, petrous_mask)
    tac_rest = extract_tac_from_path(pet_4d_path, rest_mask)
    return tac_petrous, tac_rest


def compute_auc(tac):
    return sum(tac)


def get_transform_stats(transform, description):
    try:
        if isinstance(transform, sitk.CompositeTransform):
            effective_transform = sitk.Transform(transform)
        else:
            effective_transform = transform

        params = effective_transform.GetParameters()
        fixed_params = effective_transform.GetFixedParameters()

        tx, ty, tz = params[3], params[4], params[5]
        total_translation = np.sqrt(tx**2 + ty**2 + tz**2)

        rot_deg = [np.degrees(r) for r in params[0:3]]
        total_rot_mag = np.sqrt(sum(d**2 for d in rot_deg))

        print(f"--- {description} ---")
        print(f"  Rotation (deg):  X:{rot_deg[0]:.3f}, Y:{rot_deg[1]:.3f}, Z:{rot_deg[2]:.3f} (Total: {total_rot_mag:.3f})")
        print(f"  Translation (mm): X:{tx:.3f}, Y:{ty:.3f}, Z:{tz:.3f} (Total: {total_translation:.3f}mm)")
        print(f"  Rotation Center:  {fixed_params[0]:.1f}, {fixed_params[1]:.1f}, {fixed_params[2]:.1f}")
        print("-" * (len(description) + 8))

        return {
            "Method": description,
            "Total_Shift_mm": round(total_translation, 4),
            "Rotation_Mag_deg": round(total_rot_mag, 4),
            "TX": round(tx, 4), "TY": round(ty, 4), "TZ": round(tz, 4),
            "RX_deg": round(rot_deg[0], 4), "RY_deg": round(rot_deg[1], 4), "RZ_deg": round(rot_deg[2], 4)
        }
    except Exception as e:
        print(f"Error processing {description}: {e}")
        return None


def define_registration(multi_level=[4, 2, 1], smoothing_sigmas=[2, 1, 0], use_random=False,
                         sampling_percentage=0.05, metric_bins=50):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=metric_bins)
    if use_random:
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(sampling_percentage, seed=42)
    else:
        registration_method.SetMetricSamplingStrategy(registration_method.NONE)
    if multi_level is not None:
        registration_method.SetShrinkFactorsPerLevel(multi_level)
        registration_method.SetSmoothingSigmasPerLevel(smoothing_sigmas)
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescentLineSearch(
        learningRate=0.5, numberOfIterations=100,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    return registration_method


def compute_pet_sum(pet_4d_path, output_path, n_frames=None):
    if n_frames is None:
        n_frames = cfg.N_FRAMES
    pet_4d = sitk.ReadImage(pet_4d_path, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(pet_4d)
    sum_arr = arr[0:n_frames].sum(axis=0).astype(np.float32)

    pet_sum = sitk.GetImageFromArray(sum_arr)
    pet_sum.SetOrigin(pet_4d.GetOrigin()[:3])
    pet_sum.SetSpacing(pet_4d.GetSpacing()[:3])
    dir_4d = np.array(pet_4d.GetDirection()).reshape(4, 4)
    pet_sum.SetDirection(dir_4d[:3, :3].flatten().tolist())

    sitk.WriteImage(pet_sum, output_path)
    return pet_sum


def run_rigid_registration(fixed, moving, roi_mask=None):
    #defaulted to single-threaded for reproducibility (even though it may be slower)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    p = cfg.REFINE_REGISTRATION_DEFAULTS
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=p["metric_bins"])
    if roi_mask is not None:
        reg.SetMetricSamplingStrategy(reg.NONE)
        reg.SetMetricFixedMask(roi_mask)
    else:
        reg.SetMetricSamplingStrategy(reg.REGULAR)
        # fixed seed for reproducibility 
        reg.SetMetricSamplingPercentage(p["sampling_percentage"], seed=42)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=p["learning_rate"],
        numberOfIterations=p["num_iterations"],
        convergenceMinimumValue=p["convergence_min_value"],
        convergenceWindowSize=p["convergence_window_size"],
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(sitk.Euler3DTransform(), inPlace=False)
    reg.SetShrinkFactorsPerLevel(p["shrink_factors"])
    reg.SetSmoothingSigmasPerLevel(p["smoothing_sigmas"])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    def on_iteration():
        if reg.GetOptimizerIteration() % 50 == 0:
            print(f"    iter {reg.GetOptimizerIteration():3d}  metric={reg.GetMetricValue():.6f}")

    reg.AddCommand(sitk.sitkIterationEvent, on_iteration)
    final_tx = reg.Execute(fixed, moving)
    print(f"  Stop : {reg.GetOptimizerStopConditionDescription()}")
    print(f"  Final metric : {reg.GetMetricValue():.6f}")
    return final_tx


def resample_to_reference(moving, reference, transform=None, interpolator=sitk.sitkLinear, default_value=0.0):
    """Resample moving into the space of reference using transform (identity if None)."""
    if transform is None:
        transform = sitk.Transform()
    return sitk.Resample(moving, reference, transform, interpolator, default_value, moving.GetPixelID())


def build_dilated_roi(mask_sitk, radius=None):
    if radius is None:
        radius = cfg.ROI_DILATION_RADIUS
    binary = sitk.Cast(mask_sitk > 0, sitk.sitkUInt8)
    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetKernelRadius(radius)
    dilate.SetForegroundValue(1)
    return dilate.Execute(binary)


def save_tacs_csv(tacs, out_dir, patient_id):
    csv_path = os.path.join(out_dir, f"{patient_id}_tacs.csv")
    n_frames = len(next(iter(tacs.values())))
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame"] + list(tacs.keys()))
        for t in range(n_frames):
            writer.writerow([t] + [tacs[k][t] for k in tacs])
    print(f"  Saved TACs CSV -> {csv_path}")


def get_mask_path(patient_id, method):
    patient_dir = os.path.join(cfg.OUTPUT_DIR, patient_id)
    refined_dir = os.path.join(patient_dir, "refined_segmentations")

    fixed_paths = {
        "initial_aligned_mri": os.path.join(patient_dir, "initial_aligned_mri_petrous.nii.gz"),
        "rigid_reg": os.path.join(patient_dir, "rigid_reg_petrous.nii.gz"),
        "refine_rigid_reg_petrous": os.path.join(refined_dir, "refine_rigid_reg_petrous.nii.gz"),
    }
    if method in fixed_paths:
        return fixed_paths[method]
    return os.path.join(refined_dir, f"{method}_petrous.nii.gz")


def get_file_paths_from_moco(patient_id):
    patient_base = os.path.join(cfg.BASE_DIR, patient_id)
    sub_dirs = sorted(d for d in os.listdir(patient_base) if d.startswith(cfg.MOCO_DIR_PREFIX))
    if not sub_dirs:
        raise FileNotFoundError(f"No {cfg.MOCO_DIR_PREFIX} subdirectory found for {patient_id}")
    return os.path.join(patient_base, sub_dirs[0], cfg.MOCO_SUBDIR)


def run_pvc(input_path, output_path, mask_path):
    fx, fy, fz = cfg.PVC_FWHM
    prefix = f"conda run -n {cfg.PVC_CONDA_ENV} " if cfg.PVC_CONDA_ENV else ""
    cmd = (
        f'{prefix}'
        f'petpvc -i "{input_path}" -o "{output_path}" -m "{mask_path}" '
        f'-p "{cfg.PVC_METHOD}" -x {fx} -y {fy} -z {fz}'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"petpvc failed for {input_path}: {result.stderr}")


def get_winner_method(patient_id):
    peaks = {}

    legacy_csv = os.path.join(cfg.OUTPUT_DIR, patient_id, "split_segments_peaks_summary.csv")
    if os.path.exists(legacy_csv):
        df_legacy = pd.read_csv(legacy_csv).set_index("Method")
        for method in df_legacy.index:
            peaks[method] = float(df_legacy.loc[method, "Petrous_Peak"])

    refined_csv = os.path.join(cfg.OUTPUT_DIR, patient_id, "refined_segmentations", f"{patient_id}_tacs.csv")
    if os.path.exists(refined_csv):
        df_refined = pd.read_csv(refined_csv)
        for col in df_refined.columns:
            if col == "frame":
                continue
            peaks[col] = float(df_refined[col].max())

    if not peaks:
        raise FileNotFoundError(f"No peak CSVs found for {patient_id}")

    winner = max(peaks, key=peaks.get)
    print(f"  Peak summary for {patient_id}:")
    for m, v in peaks.items():
        marker = " <-- WINNER" if m == winner else ""
        print(f"    {m}: {v:.2f}{marker}")

    return winner, peaks

def plot_tac_comparison(tac_initial, tac_identity_multi, patient_id, output_dir):
    time_points = list(range(len(tac_initial)))
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, tac_initial, label='Initial', marker='o')
    plt.plot(time_points, tac_identity_multi, label='Identity Multi-level', marker='o')
    plt.xlabel('Time Points')
    plt.ylabel('TAC Value')
    plt.title(f'TAC Comparison for {patient_id}')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"{patient_id}_tac_comparison.png"))
    plt.close()


def plot_split_tac_visualization(tacs_df, patient_id, output_dir):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))

    colors = {
        'petrous_initial': '#e74c3c',
        'petrous_rigid': '#c0392b',
        'cervical_initial': '#3498db',
        'cervical_rigid': '#2c3e50',
        'full_initial': '#27ae60',
        'full_rigid': '#16a085',
    }
    styles = {
        'initial': {'linestyle': '--', 'marker': 'o', 'markersize': 6, 'linewidth': 2.5},
        'rigid': {'linestyle': '-', 'marker': 's', 'markersize': 6, 'linewidth': 2.5},
    }

    for column in tacs_df.columns:
        col_lower = column.lower()
        if "petrous" in col_lower and "initial" in col_lower:
            color, style, label = colors['petrous_initial'], styles['initial'], 'Petrous - Initial'
        elif "petrous" in col_lower and "rigid" in col_lower:
            color, style, label = colors['petrous_rigid'], styles['rigid'], 'Petrous - Rigid'
        elif "cervical" in col_lower and "initial" in col_lower:
            color, style, label = colors['cervical_initial'], styles['initial'], 'Cervical - Initial'
        elif "cervical" in col_lower and "rigid" in col_lower:
            color, style, label = colors['cervical_rigid'], styles['rigid'], 'Cervical - Rigid'
        elif "full" in col_lower and "initial" in col_lower:
            color, style, label = colors['full_initial'], styles['initial'], 'Full - Initial'
        elif "full" in col_lower and "rigid" in col_lower:
            color, style, label = colors['full_rigid'], styles['rigid'], 'Full - Rigid'
        else:
            color, style, label = None, {'linestyle': '-', 'marker': 'x', 'markersize': 6, 'linewidth': 2}, column

        plt.plot(tacs_df.index, tacs_df[column],
                 label=label, color=color,
                 linestyle=style['linestyle'], marker=style['marker'],
                 markersize=style['markersize'], linewidth=style['linewidth'],
                 markevery=max(1, len(tacs_df) // 20), alpha=0.9)

    plt.xlabel('Time Frame', fontsize=13, fontweight='bold')
    plt.ylabel('Radioactivity Concentration (Bq/mL)', fontsize=13, fontweight='bold')
    plt.title(f'Carotid Segment Comparison: Petrous vs. Cervical vs. Full\nPatient: {patient_id}',
              fontsize=15, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0., fontsize=11, frameon=True,
               shadow=True, fancybox=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{patient_id}_split_tac_visualization.png"),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_segment_tac(tacs_df, patient_id, output_dir, keyword, title, filename_suffix):
    plt.figure(figsize=(12, 7))
    cols = [c for c in tacs_df.columns if keyword in c]
    colors_seg = {'Petrous': '#e74c3c', 'Cervical': '#3498db', 'Full': '#27ae60'}

    for col in cols:
        if "Petrous" in col:
            color, ls, m, segment = colors_seg['Petrous'], '-', 'o', 'Petrous'
        elif "Cervical" in col:
            color, ls, m, segment = colors_seg['Cervical'], '--', 's', 'Cervical'
        elif "Full" in col:
            color, ls, m, segment = colors_seg['Full'], '-.', '^', 'Full'
        else:
            color, ls, m, segment = 'gray', '-', 'x', col

        plt.plot(tacs_df.index, tacs_df[col],
                 label=segment, linestyle=ls, marker=m, markersize=5,
                 linewidth=2.5, color=color,
                 markevery=max(1, len(tacs_df) // 15), alpha=0.9)

    plt.title(f"{title} - {patient_id}", fontsize=15, fontweight='bold', pad=20)
    plt.xlabel("Time Frames", fontsize=12, fontweight='bold')
    plt.ylabel("Activity (Bq/mL)", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.margins(x=0.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{patient_id}_{filename_suffix}_tac_visualization.png"),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_refinement_comparison(tacs, patient_id, out_dir):
    print("\n  Saving comparison TAC plot...")
    frames = np.arange(len(next(iter(tacs.values()))))

    colors = {
        "identity":                     "#1f77b4",
        "initial_petrous_refined_full": "#ff7f0e",
        "initial_petrous_refined_roi":  "#9467bd",
        "refine_rigid_reg_petrous":     "#2ca02c",
        "refined_petrous_sum":          "#d62728",
    }
    labels = {
        "identity":                     "Identity (no reg)",
        "initial_petrous_refined_full": "Identity + refined (full)",
        "initial_petrous_refined_roi":  "Identity + refined (ROI)",
        "refine_rigid_reg_petrous":     "Refined reg MRI",
        "refined_petrous_sum":          "Full MRI rigid (novel)",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for key, tac in tacs.items():
        ax.plot(frames, tac, marker="o", color=colors[key], label=labels[key])

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Mean activity (a.u.)")
    ax.set_title(f"{patient_id} - petrous carotid TAC comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, f"{patient_id}_tac_comparison.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot -> {plot_path}")
