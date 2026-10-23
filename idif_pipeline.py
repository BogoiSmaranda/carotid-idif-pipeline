import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

import numpy as np
import pandas as pd
import SimpleITK as sitk

import idif_config as cfg
from idif_utils import (
    get_paths, define_registration, extract_tac, compute_auc, get_transform_stats,
    split_save_and_extract, plot_tac_comparison, plot_split_tac_visualization, plot_segment_tac,
    get_mask_path, compute_pet_sum, resample_to_reference, split_petrous_and_extract,
    run_rigid_registration, build_dilated_roi, extract_tac_from_path, save_tacs_csv,
    plot_refinement_comparison,
    get_file_paths_from_moco, get_winner_method, run_pvc, extract_tac_frame,
)

def run_initial_registration(patient_id):
    """rigid MRI->PET registration, the global one"""
    print(f"\n{'='*60}")
    print(f"Processing patient: {patient_id}")
    print(f"{'='*60}")

    output_dir = os.path.join(cfg.OUTPUT_DIR, patient_id)
    os.makedirs(output_dir, exist_ok=True)

    pet_path, mri_path, mask_path = get_paths(patient_id)

    try:
        pet_image = sitk.ReadImage(pet_path, sitk.sitkFloat32)
        pet_ref = pet_image[:, :, :, cfg.PET_REF_FRAME_INDEX]
        mri_image = sitk.ReadImage(mri_path, sitk.sitkFloat32)
        mask_image = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    except Exception as e:
        print(f"Error loading images for {patient_id}: {e}")
        return None

    # compute intial resamples mask for comparison with the rigid registration
    initial_transform = sitk.Euler3DTransform()

    mask_resampler = sitk.ResampleImageFilter()
    mask_resampler.SetReferenceImage(pet_ref)
    mask_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_resampler.SetTransform(initial_transform)
    initial_resampled_mask = mask_resampler.Execute(mask_image)
    sitk.WriteImage(initial_resampled_mask,
                    os.path.join(output_dir, f"{patient_id}_initial_resampled_mask_original.nii.gz"))

    # execute rigid registration with default settings
    print("Starting registrations with different settings...")
    print("Identity Multi-level")

    registration_method_1 = define_registration(**cfg.REGISTRATION_DEFAULTS)
    registration_method_1.SetInitialTransform(initial_transform, inPlace=False)
    final_transform_identity_multi = registration_method_1.Execute(pet_ref, mri_image)

    registered_mri_identity_multi = sitk.Resample(
        mri_image, pet_ref, final_transform_identity_multi, sitk.sitkLinear, 0.0, mri_image.GetPixelID())
    mask_registered_mri_identity_multi = sitk.Resample(
        mask_image, pet_ref, final_transform_identity_multi, sitk.sitkNearestNeighbor, 0.0, mask_image.GetPixelID())

    sitk.WriteImage(registered_mri_identity_multi, os.path.join(output_dir, f"{patient_id}_registered_mri.nii.gz"))
    sitk.WriteImage(mask_registered_mri_identity_multi, os.path.join(output_dir, f"{patient_id}_registered_mask.nii.gz"))

    # extract TACs for both initial and rigid registration 
    tac_initial = extract_tac(pet_image, initial_resampled_mask)
    tac_identity_multi = extract_tac(pet_image, mask_registered_mri_identity_multi)

    tac_df = pd.DataFrame({
        "Initial": tac_initial,
        "Identity_Multi": tac_identity_multi,
    })
    tac_df.to_csv(os.path.join(output_dir, f"{patient_id}_tac_comparison.csv"), index=False)
    plot_tac_comparison(tac_initial, tac_identity_multi, patient_id, output_dir)

    auc_initial = compute_auc(tac_initial)
    auc_identity_multi = compute_auc(tac_identity_multi)
    auc_df = pd.DataFrame({
        "Method": ["Initial", "Identity Multi-level"],
        "AUC": [auc_initial, auc_identity_multi],
    })
    auc_df.to_csv(os.path.join(output_dir, f"{patient_id}_auc_comparison.csv"), index=False)
    print("AUC Comparison: ")
    print(auc_df)

    print("Compute the peaks")
    peak_initial = max(tac_initial)
    peak_identity_multi = max(tac_identity_multi)
    peak_df = pd.DataFrame({
        "Method": ["Initial", "Identity Multi-level"],
        "Peak": [peak_initial, peak_identity_multi],
    })
    peak_df.to_csv(os.path.join(output_dir, f"{patient_id}_peak_comparison.csv"), index=False)
    print("Peak Comparison: ")
    print(peak_df)

    # save the transformation parameters for displacement analysis if needed
    registration_results = []
    for tf, desc in [(final_transform_identity_multi, "Identity Multi-level")]:
        stats = get_transform_stats(tf, desc)
        if stats:
            registration_results.append(stats)

    dist_df = pd.DataFrame(registration_results)
    dist_df.to_csv(os.path.join(output_dir, f"{patient_id}_registration_displacement.csv"), index=False)
    print("\nFinal Displacement Summary Saved to CSV:")
    print(dist_df[["Method", "Total_Shift_mm", "Rotation_Mag_deg"]])

    mask_dict = {
        "initial_aligned_mri": initial_resampled_mask,
        "rigid_reg": mask_registered_mri_identity_multi,
    }

    final_split_stats = []
    all_tacs_data = {}

    for name, mask in mask_dict.items():
        print(f"Splitting, Saving Masks, and Extracting TACs: {name}")
        t_pet, t_cer = split_save_and_extract(pet_image, mask, name, output_dir, buffer_slices=cfg.BUFFER_SLICES)

        if t_pet is not None:
            final_split_stats.append({
                "Method": name,
                "Petrous_Peak": np.max(t_pet),
                "Cervical_Peak": np.max(t_cer) if len(t_cer) > 0 else 0,
            })
            all_tacs_data[f"{name}_Petrous"] = t_pet
            all_tacs_data[f"{name}_Cervical"] = t_cer
            all_tacs_data[f"{name}_Full"] = extract_tac(pet_image, mask)

    tacs_df = pd.DataFrame(all_tacs_data)
    tacs_df.to_csv(os.path.join(output_dir, f"{patient_id}_full_split_tacs.csv"), index_label="Frame")

    split_summary_df = pd.DataFrame(final_split_stats)
    split_summary_df.to_csv(os.path.join(output_dir, "split_segments_peaks_summary.csv"), index=False)

    print("\nSaved Full TACs to CSV. Columns available:")
    print(tacs_df.columns.tolist())

    plot_split_tac_visualization(tacs_df, patient_id, output_dir)
    plot_segment_tac(tacs_df, patient_id, output_dir, "initial", "No Registration", "no_registration")
    plot_segment_tac(tacs_df, patient_id, output_dir, "rigid", "Rigid Registration", "rigid_registration")

    print(f"\nCompleted processing for {patient_id}")

    return {
        'patient_id': patient_id,
        'auc_initial': auc_initial,
        'auc_identity_multi': auc_identity_multi,
        'peak_initial': peak_initial,
        'peak_identity_multi': peak_identity_multi,
        'total_shift_mm': dist_df['Total_Shift_mm'].values[0] if len(dist_df) > 0 else None,
        'rotation_mag_deg': dist_df['Rotation_Mag_deg'].values[0] if len(dist_df) > 0 else None,
    }



def run_carotid_refinement(patient_id):
    """Refine the carotid petrous mask using 5 candidate methods
    The PET sum (first 18 frames, 0-3 min) is the co-registration target.
    Method 1: identity resample of original MRI and segmentation to PET sum space (no registration, baseline)
    Method 2 (a): PET-space global rigid registration: refine identity resample with rigid registration (global registration) to the PET sum
    Method 3 (b): PET-space VOI-based rigid registration: refine identity resample with rigid registration to the PET sum, only on a dilated carotid VOI
    Method 4 (c): VOI-based rigid refinement: refine the initial rigid registration mask with a VOI-based registration to the PET sum
    Method 5 (d): native-space global rigid registration: full original MRI rigid reg to PET sum, no prior resampling
    """
    print(f"\n{'='*60}")
    print(f"Patient: {patient_id}")
    print(f"{'='*60}")

    patient_dir = os.path.join(cfg.OUTPUT_DIR, patient_id)
    out_dir = os.path.join(patient_dir, "refined_segmentations")
    os.makedirs(out_dir, exist_ok=True)

    pet_4d_path, mri_orig_path, seg_orig_path = get_paths(patient_id)
    # get rigid registered MRI and petrous mask from run_initial_registration
    mri_reg_path = os.path.join(patient_dir, f"{patient_id}_registered_mri.nii.gz")
    petrous_reg_path = get_mask_path(patient_id, "rigid_reg")
    pet_sum_path = os.path.join(out_dir, f"{patient_id}_PET_sum_first18.nii.gz")

    print("  Computing PET sum (frames 0-17)...")
    pet_sum = compute_pet_sum(pet_4d_path, pet_sum_path)

    mri_orig = sitk.ReadImage(mri_orig_path, sitk.sitkFloat32)
    seg_orig = sitk.ReadImage(seg_orig_path, sitk.sitkUInt8)

    tacs = {}

    # Method 1: just resample the original MRI and segmentation to PET space (identity transform)
    print("\n  [1] Identity resample...")
    identity_tx = sitk.Transform()
    sitk.WriteTransform(identity_tx, os.path.join(out_dir, "identity_transform.tfm"))
    mri_identity = resample_to_reference(mri_orig, pet_sum)
    seg_identity = resample_to_reference(seg_orig, pet_sum,
                                          interpolator=sitk.sitkNearestNeighbor,
                                          default_value=0.0)
    sitk.WriteImage(mri_identity, os.path.join(out_dir, "identity_mri.nii.gz"))

    tac_petrous_id, _ = split_petrous_and_extract(pet_4d_path, seg_identity, "identity", out_dir)
    tacs["identity"] = tac_petrous_id

    # Method 2 (a): PET-space global rigid registration: refine the identity resample with rigid registration to the PET sum
    print("\n  [2] Refining identity — full image (no ROI mask)...")
    tx_identity_refined_full = run_rigid_registration(pet_sum, mri_identity)
    sitk.WriteTransform(tx_identity_refined_full,
                        os.path.join(out_dir, "identity_refined_full_transform.tfm"))

    seg_identity_refined_full = sitk.Resample(
        seg_identity, pet_sum, tx_identity_refined_full,
        sitk.sitkNearestNeighbor, 0, seg_identity.GetPixelID())
    tac_petrous_idr_full, _ = split_petrous_and_extract(
        pet_4d_path, seg_identity_refined_full, "initial_petrous_refined_full", out_dir)
    tacs["initial_petrous_refined_full"] = tac_petrous_idr_full

    # Method 3 (b): PET-space VOI-based rigid registration: as (a), but only on the dilated carotid VOI
    print("\n  [3] Refining identity — dilated petrous ROI mask...")
    roi_mask_identity = build_dilated_roi(seg_identity)
    tx_identity_refined_roi = run_rigid_registration(pet_sum, mri_identity, roi_mask=roi_mask_identity)
    sitk.WriteTransform(tx_identity_refined_roi,
                        os.path.join(out_dir, "identity_refined_roi_transform.tfm"))

    seg_identity_refined_roi = sitk.Resample(
        seg_identity, pet_sum, tx_identity_refined_roi,
        sitk.sitkNearestNeighbor, 0, seg_identity.GetPixelID())
    tac_petrous_idr_roi, _ = split_petrous_and_extract(
        pet_4d_path, seg_identity_refined_roi, "initial_petrous_refined_roi", out_dir)
    tacs["initial_petrous_refined_roi"] = tac_petrous_idr_roi

    # Method 4 (c): VOI-based rigid refinement: refine the initial registration mask with a VOI-based registration to the PET sum
    print("\n  [4] Refining registered MRI (existing method)...")
    mri_reg = sitk.ReadImage(mri_reg_path, sitk.sitkFloat32)
    petrous_reg = sitk.ReadImage(petrous_reg_path, sitk.sitkUInt8)

    roi_mask = build_dilated_roi(petrous_reg)
    tx_refine = run_rigid_registration(pet_sum, mri_reg, roi_mask=roi_mask)
    sitk.WriteTransform(tx_refine, os.path.join(out_dir, "refine_rigid_reg_transform.tfm"))

    petrous_refined = sitk.Resample(
        petrous_reg, pet_sum, tx_refine,
        sitk.sitkNearestNeighbor, 0, petrous_reg.GetPixelID())
    sitk.WriteImage(petrous_refined, os.path.join(out_dir, "refine_rigid_reg_petrous.nii.gz"))
    tacs["refine_rigid_reg_petrous"] = extract_tac_from_path(pet_4d_path, petrous_refined)

    # Method 5 (d): native-space global rigid registration: MRI to PET sum from scratch (no prior resampling) using the original MRI and segmentation
    print("\n  [5] Full original MRI rigid reg to PET sum...")
    tx_full = run_rigid_registration(pet_sum, mri_orig)
    sitk.WriteTransform(tx_full, os.path.join(out_dir, "refined_petrous_sum_transform.tfm"))

    seg_full_refined = sitk.Resample(
        seg_orig, pet_sum, tx_full,
        sitk.sitkNearestNeighbor, 0, seg_orig.GetPixelID())
    tac_petrous_full, _ = split_petrous_and_extract(
        pet_4d_path, seg_full_refined, "refined_petrous_sum", out_dir)
    tacs["refined_petrous_sum"] = tac_petrous_full

    save_tacs_csv(tacs, out_dir, patient_id)
    plot_refinement_comparison(tacs, patient_id, out_dir)

    return patient_id, None


def run_final_tacs(patient_id):
    """Pick the winning carotid mask (based on peak values), run PVC, extract final TACs."""
    print(f"\n{'='*60}")
    print(f"Processing patient: {patient_id}")
    print(f"{'='*60}")

    patient_dir = os.path.join(cfg.OUTPUT_DIR, patient_id)
    pet_orig_path, _, _ = get_paths(patient_id)
    moco_dir = get_file_paths_from_moco(patient_id)
    last_frame = cfg.LAST_FRAME_OVERRIDES.get(patient_id, cfg.DEFAULT_LAST_FRAME)

    winner, peaks = get_winner_method(patient_id)
    winner_mask_path = get_mask_path(patient_id, winner)
    print(f"  Winner: {winner}  ->  {winner_mask_path}")

    winner_mask = sitk.ReadImage(winner_mask_path, sitk.sitkUInt8)

    print("  Running PVC on winner mask (parallel)...")
    pvc_dir = os.path.join(patient_dir, cfg.PVC_DIR_NAME)
    os.makedirs(pvc_dir, exist_ok=True)

    pvc_jobs = []
    for i in range(0, cfg.N_FRAMES):
        inp = os.path.join(moco_dir, cfg.NO_MOCO_VOLUME_PATTERN.format(i=i))
        out = os.path.join(pvc_dir, f"pvc_winner_frame_{i}.nii.gz")
        pvc_jobs.append((i, inp, out))
    for i in range(cfg.N_FRAMES, last_frame):
        inp = os.path.join(moco_dir, cfg.MOCO_VOLUME_PATTERN.format(i=i))
        out = os.path.join(pvc_dir, f"pvc_winner_frame_{i}.nii.gz")
        pvc_jobs.append((i, inp, out))

    with ThreadPoolExecutor(max_workers=cfg.PVC_THREADS) as executor:
        futures = {
            executor.submit(run_pvc, inp, out, winner_mask_path): idx
            for idx, inp, out in pvc_jobs
        }
        for future in as_completed(futures):
            idx = futures[future]
            future.result()  # raises on failure
            print(f"  PVC frame {idx} done")

    print("  Extracting TACs...")
    pet_volume = sitk.ReadImage(pet_orig_path)

    tac_raw, tac_moco, tac_pvc = [], [], []

    for i in range(0, cfg.N_FRAMES):
        frame_raw = pet_volume[:, :, :, i]
        tac_raw.append(extract_tac_frame(frame_raw, winner_mask))
        tac_moco.append(tac_raw[-1])  # frames before N_FRAMES: no moco, same as raw
        tac_pvc.append(extract_tac_frame(
            sitk.ReadImage(os.path.join(pvc_dir, f"pvc_winner_frame_{i}.nii.gz")),
            winner_mask
        ))

    for i in range(cfg.N_FRAMES, last_frame):
        frame_raw = pet_volume[:, :, :, i]
        frame_moco = sitk.ReadImage(os.path.join(moco_dir, cfg.MOCO_VOLUME_PATTERN.format(i=i)))
        tac_raw.append(extract_tac_frame(frame_raw, winner_mask))
        tac_moco.append(extract_tac_frame(frame_moco, winner_mask))
        tac_pvc.append(extract_tac_frame(
            sitk.ReadImage(os.path.join(pvc_dir, f"pvc_winner_frame_{i}.nii.gz")),
            winner_mask
        ))

    n_frames = len(tac_raw)
    rows = []
    for t in range(n_frames):
        rows.append({
            "frame": t,
            "winner_method": winner,
            "raw": tac_raw[t],
            "moco": tac_moco[t],
            "moco_pvc": tac_pvc[t],
        })
    df = pd.DataFrame(rows)

    peaks_out = os.path.join(patient_dir, "winner_peak_summary.csv")
    pd.DataFrame([{"method": m, "peak": v} for m, v in peaks.items()]).to_csv(peaks_out, index=False)

    csv_out = os.path.join(patient_dir, f"{patient_id}_tac_refined_winner.csv")
    df.to_csv(csv_out, index=False)
    print(f"  Saved TAC CSV -> {csv_out}")
    print(f"  Saved peak summary -> {peaks_out}")

    return patient_id, None



def run_full_pipeline(patient_id):
    # Run the full pipeline, but also can comment out parts if needed for debugging
    try:
        run_initial_registration(patient_id)
        run_carotid_refinement(patient_id)
        run_final_tacs(patient_id)
        return patient_id, None
    except Exception as e:
        print(f"  ERROR for {patient_id}: {e}")
        return patient_id, str(e)


def run_pipeline(patient_ids=None, n_workers=None):
    patient_ids = patient_ids or cfg.PATIENT_IDS
    n_workers = n_workers or cfg.N_WORKERS

    with Pool(processes=n_workers) as pool:
        results = pool.map(run_full_pipeline, patient_ids)

    failed = [(pid, err) for pid, err in results if err is not None]
    print(f"\n{'='*60}")
    print(f"Done. {len(patient_ids) - len(failed)}/{len(patient_ids)} succeeded.")
    if failed:
        for pid, err in failed:
            print(f"  FAILED {pid}: {err}")
    return results


if __name__ == "__main__":
    run_pipeline()
