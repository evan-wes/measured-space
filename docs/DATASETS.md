# Datasets

Survey date: 2026-10-26. Tags: **[V]** checked against an official page, file or API (source linked);
**[U]** inferred, read from third-party code, or not checked. Sizes are bytes from the Hugging Face
tree API or HTTP `Content-Length`; GB = 10^9 bytes. No dataset payloads were downloaded for this survey.

Raw dataset files are never committed to this repository. Only code, configs, small derived
summaries and figures are committed, and only where the license allows it (see each section).

## Summary

Selection (decided 2026-10-26): the prototype shows the *feasibility* of the audit on realistic data. Each
dataset stands in for a capture setup common in deployed perception systems: handheld/egocentric RGB-D with
inertial poses, and fixed, calibrated multi-camera installations.

| Dataset | Role here | Real / synthetic | License | Derived-data redistribution | Planned download |
|---|---|---|---|---|---|
| ARKitScenes (3DOD + upsampling) | **Headline real-sensor result.** Handheld RGB + LiDAR depth + VIO poses (the egocentric RGB-D + IMU capture setup); homes | Real (iPad LiDAR + Faro laser) | Apple custom license (2024) | Unclear — commit aggregates only | 0.43 GB (6 laser scenes, self-contained) |
| NVIDIA Physical AI Smart Spaces (2026 split) | **Fixed multi-camera setting** (calibrated ceiling cameras, as in video-analytics deployments); exact synthetic GT, plus 2 real scenes with no GT | Synthetic (+2 real, no GT) | CC-BY-4.0 | Allowed with attribution | ~4.35 GB |
| Q-Spatial++ | Real photos with **physically measured** distances; VLM scoring only | Real | Apache-2.0 | Allowed | 0.13 GB |
| LOCO | Real logistics images, 2D boxes: tests the detector stage on real warehouse imagery | Real | CC0-1.0 | Allowed | 0.77 GB |
| NVIDIA SDG-Warehouse | Not used; 6-month candidate (eye-level views, maybe racks/pallets) | Synthetic | OpenMDW-1.1 | Allowed | None |
| NVIDIA PhysicalAI-Spatial-Intelligence-Warehouse | Closest prior work; cite | Synthetic | CC-BY-4.0 (+ Llama 3.1 terms for its rephrased text) | Allowed with attribution | None |
| MMPTRACK | Reviewed, **not used** (§5) | Real (staged lab sets) | Microsoft research-only agreement | No (no derived datasets) | None |

## 1. NVIDIA Physical AI Smart Spaces (`nvidia/PhysicalAI-SmartSpaces`)

- Card: https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces
- Paper [V]: Wang et al., *Physical AI Smart Spaces: A Large-Scale Benchmark for Multi-Camera 3D Perception in Smart Spaces*, arXiv:2610.02580 (submitted 2026-10-01), NeurIPS 2026 Evaluations & Datasets track. It is a multi-camera 3D **tracking** benchmark; it is not the source of any spatial QA.
- **License [V]: CC-BY-4.0.** Derived data may be redistributed with attribution. The card says the data is not authorized for biometric or surveillance use on real people; that does not affect this project.
- **Real vs synthetic [V]:** Isaac Sim synthetic. In 2026, scenes 001/003/005/007/021/024 are Cosmos-Transfer re-renderings of 000/002/004/006/020/023 (same geometry and GT, different RGB; single-view, not multi-view consistent). Test scenes `Warehouse_026` and `Warehouse_027` are real, with video and calibration only.
- **Layout [V]:** `MTMC_Tracking_{2024,2025,2026}/{train,val,test}/<Scene>/` with `calibration.json`, `ground_truth.json`, `map.png`, `videos/Camera_XXXX.mp4`, `depth_maps/Camera_XXXX.h5`. 2024 has no depth and no 3D boxes.
- **Coverage [V]:** 2026 has `Warehouse_000–027`. Train and val (000–022) have GT and depth; test (023–027) has neither. In 2025, test scenes have depth but no GT.
- **Classes [V]:** Person, Forklift, NovaCarter, Transporter, FourierGR1T2, AgilityDigit, plus PalletTruck in 2026. **Racks, shelves, pallets and boxes have no ground truth.**
- **Video [V]:** 1080p, 30 FPS, GT per frame.
- **Calibration [V]**, read from one 7 KB `calibration.json`: per camera `intrinsicMatrix` (example: fx = fy = 916.25, cx = 960, cy = 540, no distortion), `extrinsicMatrix` 3×4 (world→camera, metres, z-up world), `cameraMatrix`, `homography`. Camera centres computed as −Rᵀt match the listed coordinates. The cameras are ceiling CCTV about 3.3 m up.
- **3D boxes [V card; U conventions]:** object centre in world coordinates, `scale [w, l, h]`, `rotation [pitch, roll, yaw]`, plus `2d_bounding_box_visible {camera_id: [xmin, ymin, xmax, ymax]}`, which gives per-camera visibility directly. NVIDIA's loader converts rotations with `np.radians` (so the file stores degrees) and uses keys with spaces rather than underscores. The challenge page says radians. **Superseded: measured as radians; see "Conventions measured" below.**
- **Depth [V format; U encoding]:** one HDF5 file per camera (superseded: frames are raw gzip uint16 arrays, not PNG; see "Conventions measured" below). Third-party code reads keys `distance_to_image_plane_{frame:05d}.png` in millimetres (planar z-depth). The files are served with HTTP range support, so reading single frames remotely may be possible [U, untested].
- **Size [V, tree API]:** 2026 split 2.478 TB (49.4 GB without depth). The card says 2.94 TB. Per-scene depth ranges from 16 to 310 GB.
- **Minimal subset [V sizes, re-checked 2026-10-26]:** `MTMC_Tracking_2026/train/Warehouse_012` (8 cameras, not a Cosmos re-render), full scene 30.377 GB:
  - Stage 1, 0.152 GB: `calibration.json`, `ground_truth.json` (151.8 MB), `map.png`. Use it to choose cameras by visible-object counts.
  - Stage 2, about 4.1 GB: 2 chosen cameras' `videos/*.mp4` (94–160 MB each) and `depth_maps/*.h5` (1.4–6.8 GB each). Cameras 0004 and 0006 are the smallest pair (4.06 GB including stage 1), but cameras should be chosen by content.
  - Real scenes `MTMC_Tracking_2026/test/Warehouse_026` (4 cameras) and `Warehouse_027` (7 cameras): `calibration.json`, `vggt_reprojection_error.txt` and 60 s MP4s per camera; **no GT, no depth**; 0.290 GB together [V, tree API, survey of 2026-10-26]. Used only for a qualitative sim→real check of the auto-labeller and gate.
- **Conventions measured on `Warehouse_018` (2026-10-26; `scripts/check_smartspaces_oracle.py`):**
  - **Extrinsic:** world→camera. −Rᵀt reproduces each sensor's `coordinates` in all 4 local scenes; read the
    other way, projected-box IoU is 0.
  - **Rotation:** **radians**, `[pitch, roll, yaw]`, yaw counter-clockwise about +z (yaw spans ±π; pitch and
    roll ≈ 0). Reading it as degrees scores worse on both reprojection IoU and depth containment.
  - **Scale:** `(width, length, height)` along the box's local x, y, z. Forklifts travel along local −y.
  - **Depth:** planar z in millimetres, stored as raw gzip-compressed uint16 arrays (1080×1920) despite the
    `.png` key names. Back-projected as planar depth, the floor lands at z = 0.002 m (median).
  - **2D boxes:** visible boxes are inclusive pixel indices.
  - **Video:** OpenCV frame seeking matches sequential decoding.
  - **Principal point:** the intrinsics put it at exactly (W/2, H/2), suggesting a corner-origin pixel
    convention (a 0.5 px offset from this package's centre-origin convention; not corrected).
- **`Warehouse_018` contents:** 4 objects, Forklift_3 (parked), Forklift_4 (moving), PalletTruck_2 (parked) and
  Person_0; 9,000 frames. Ceiling cameras 0000/0002 are 3.26/3.21 m above the floor. Every GT visible 2D box lies
  inside its projected 3D box (median covered share 1.000 for every class and camera). Open: Forklift_4's box
  bottom sits 0.14–0.16 m above the floor, unlike the others.
- **Additional scenes for diversity (2026-10-26).** Warehouse_018 alone is one scene with 4 objects. Nine
  more 2026 scenes containing forklifts, pallet trucks and people were added **without depth**: train
  000/002/004/006/010/011/017/019 and val 020. Their questions use the GT's per-camera visible boxes for
  visibility, and their automatic labels use monocular depth or the calibrated ground plane.
  - Per scene: `calibration.json` + `ground_truth.json` (1.7 GB for all 9), then two cameras' videos (3.31 GB
    for the 18 videos).
  - The two cameras per scene were chosen from the GT alone by `scripts/select_smartspaces_cameras.py`: the pair
    seeing the most objects in common, among pairs that see a forklift and a person together in ≥50% of
    sampled frames.
  - Result in `configs/smartspaces_cameras.yaml`: 3.3–24.9 shared objects per frame; joint forklift+person
    presence 0.61–1.00.
- **Further scenes (2026-10-26, second expansion).** 8 more independent 2026 scenes (train 008/009/012/013/014/015/016,
  val 022; only 016 has forklifts) plus the 5 Cosmos-Transfer re-renders (001/003/005/007/021). The re-renders
  share geometry, GT and **identical calibration** with their sources (000/002/004/006/020; checked). They use
  the same two cameras as their sources, so they form a paired **appearance-shift** test, not independent scenes.
  - Cameras for the independent scenes were chosen by the same script. Scenes without forklifts fall back to
    the most co-visible pair, 2.6–30.3 shared objects per frame.
  - Downloads: 2.28 GB of GT (+ small calibration files) for 12 of these scenes (Warehouse_012's was already local) and 3.60 GB
    of video (26 videos).
  - The warehouse side is now 18 independent scenes (Warehouse_018 with depth + 17 depth-less) plus 5
    appearance variants.
- **Question-type mapping:** all planned types are supported for the 7 agent classes. These are centre and closest-surface distance, size, egocentric distance, relative depth, left/right, closest same-class instance, and count in view (from `2d_bounding_box_visible`).
- **Risks:**
  - An open-vocabulary detector will find racks and pallets that have no GT; those detections cannot be audited and must be excluded from scoring.
  - Robot class names are hard text prompts.
  - Views are high CCTV only, so people are small in frame.
  - GT key spelling and degrees vs radians are unconfirmed.
  - Sim-to-real gap.

## 2. ARKitScenes (Apple)

- Repo: https://github.com/apple/ARKitScenes. Paper: arXiv:2111.08897.
- **License [V]: the brief's "CC BY-NC-SA 4.0" is out of date.** In 2024 the README dropped the CC statement and pointed to `LICENSE`, now an Apple software license. It grants (a) a personal, non-commercial license and (b) commercial terms for licensees under 700M monthly active users. Both cover use, reproduction, modification and redistribution "with or without modifications", on condition that the notice is kept and Apple marks are not used. Because it is written for "Apple Software", it is **unclear** how it applies to derived data (contact: ARKitScenes-license@group.apple.com). **Plan: commit only aggregate statistics and figures for ARKitScenes, not per-item QA files, unless you decide otherwise.** Omni3D still bundles the old CC BY-NC-SA text.
- **Domain [V]:** real homes; iPad Pro LiDAR plus a Faro laser scanner.
- **Self-contained copy (decided 2026-10-26):** this project uses only its own downloads, not a raw-subset copy
  elsewhere on the machine. For the 6 validation videos with laser depth, each `upsampling/Validation/<id>.zip`
  holds 31/19/14/23/31/31 filtered frames (149 in total, 148 with an interpolable pose), each with `wide` (1920×1440 RGB), `highres_depth` (Faro laser), `lowres_depth`
  (iPad LiDAR) and `confidence`. Checked on 42444946: 31 files per folder. Alongside the zips we downloaded
  `wide_intrinsics.zip` (per-frame `.pincam`), `<id>_3dod_annotation.json` and `lowres_wide.traj` (ARKit poses,
  10 Hz; 539,358 B for all 6, fetched 2026-10-26). Poses must be interpolated to each `wide` frame's timestamp.
  The 4 videos without laser depth are not used.
- **Those 10 videos [V, Apple `raw/metadata.csv`]:** all 10 have `is_in_threedod=True` and laser point clouds. **Six have laser `highres_depth`** (`is_in_upsampling=True`): 42444946, 42444966, 42445021, 42445028, 42445429, 42446100. `sky_direction` is Left ×6, Down ×3, Up ×1, so frames must be rotated upright before VLM use, with left/right defined in the upright image.
- **Boxes [V]:** 17 furniture classes, oriented (`obbAligned` centroid, axesLengths (full size), normalizedAxes). Human-labelled; no published accuracy.
  - Gravity-aligned / yaw-only [V, measured]: the world is z-up and every box is yaw-only except two in 42444946
    (tilted 9.47°) and one in 42445028 (2.21°).
  - Coordinate frame [V, measured with `scripts/check_arkitscenes_oracle.py`]: boxes share the frame of
    `lowres_wide.traj`.
    - Trajectory rows are world→camera and are inverted, with OpenCV camera axes (no ARKit axis flip).
    - Box rotation = `reshape(normalizedAxes, (3,3)).T` (the stored rows are the box axes).
    - No time offset is needed: the fit peaks at 0 s within ±0.02 s.
    - Over all 148 posed frames, the boxes explain a median 0.75 of the laser points in their ray region. In-box
      laser points lie a median 3.8 cm from the box surface (90th percentile 16.4 cm; objects are not boxes).
      The rejected conventions explain 0.00–0.54.
  - **Object ids:** `objectId` repeats within a file; the loader uses `uid`.
- **Expansion (2026-10-26).** Six videos left the per-label error model with too little data, so 101 more
  validation videos with laser depth and 3D boxes were added: one per venue, from the 285 validation videos
  that have both, excluding the 6 used venues, in a seeded random order.
  - Same files per video: upsampling zip, 3DOD annotation, wide intrinsics, `lowres_wide.traj`.
  - All 107 zips pass a CRC check, 5.4 GB in total.
  - Across all folds, 2,233 videos (837 venues) have both laser depth and 3D boxes (Apple `raw/metadata.csv`).
  - **Image orientation:** the rotation to upright is derived from gravity (pose), not the `sky_direction`
    text. It agrees with the metadata on all 6 videos ("Left" = turn 90° clockwise, "Down" = 180°).
  - **Laser vs LiDAR:** laser depth reads a median 2.1 cm deeper than iPad LiDAR at confidence-2 pixels (+1.7
    to +2.3 cm per video; 6.15 M pixels). This is a small systematic offset for the error model.
  - **Coverage:** 10 of the 78 boxes are never visible in a laser frame (6 never in frustum, 4 behind nearer
    surfaces).
- **Incremental download [V, HEAD sizes]:**
  - 10 × `<id>_3dod_annotation.json`: 156,558 bytes in total.
  - 6 × `upsampling/Validation/<id>.zip`: 422.7 MB (1920×1440 colour, laser `highres_depth`, lowres depth, confidence). Laser depth is aligned to the `wide` timestamps, not to `vga_wide`.
  - Optional: 10 × `<id>_3dod_mesh.ply`, 355.5 MB.
- **Question-type mapping:** distance, size, egocentric distance, relative depth, left/right and same-class closest are all supported. Count in view is **partial**: boxes are per scene, so visibility needs a frustum test plus depth or mesh occlusion.

## 3. NVIDIA SDG-Warehouse (`nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes`) — contender, not yet probed

- **License [V]:** OpenMDW-1.1. Redistribution is unrestricted if notices are kept, and outputs carry no obligations. Not gated.
- **Data [V card]:**
  - Isaac Sim warehouse with shelves and props, 4 scenarios, 5–10 synchronized cameras per run, including eye-level cameras.
  - Per frame, `object_detection.jsonl` has oriented 3D boxes of "agents and props" in world coordinates, and `camera_params.jsonl` has intrinsics and extrinsics.
  - Lossless per-pixel instance-ID NPZ files give exact visibility.
  - **Depth is a log-normalized colorized H.264 MP4 (lossy).**
- **Unverified [U]:** whether shelves, pallets and boxes are among the labelled classes. This decides whether it beats Smart Spaces on question coverage.
- **Size [V]:** 18.57 TiB total. Probe recipe, about 0.67 GB: `metadata/runs.parquet`, `metadata/clips.parquet`, `artifacts/forklift_shelf_collision/forklift_collision-artifacts-00259.tar`.

## 4. NVIDIA PhysicalAI-Spatial-Intelligence-Warehouse — cite, do not use as oracle

- Card: https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse; AI City Challenge 2025 Track 3; overview arXiv:2508.13564.
- **License [V]:** CC-BY-4.0; text rephrased by Llama-3.1-70B is also under the Llama 3.1 Community License. Gated (automatic approval).
- **Content [V]:** synthetic warehouse RGB-D, about 95k images, with 499k/1.9k/19k train/val/test QA pairs. Categories: left_right, multiple choice, distance (metres), count. Answers are computed from simulator GT; the distance definition is not documented. Scored by Acc@10 / exact match. Top leaderboard score 95.86.
- **Why not the oracle [V]:** it ships **no 3D boxes and no camera intrinsics**, so labels can be neither regenerated nor audited. Size and egocentric questions are impossible. Depth appears to be 8-bit normalized [U].
- Total size 261.19 GB; val split 5.611 GB (optional external benchmark).

## 5. Q-Spatial++ (`andrewliao11/Q-Spatial-Bench`)

- Card: https://huggingface.co/datasets/andrewliao11/Q-Spatial-Bench; paper arXiv:2409.09788 (Liao et al.).
- **License [V, HF API]:** Apache-2.0, not gated. Derived data may be redistributed.
- **Content [V]:** 101 questions on 87 real photos with distances "physically measured" (the measuring tool is not stated). No calibration or 3D boxes [U].
- **Size [V, tree API]:** 129.4 MB in total; `data/QSpatial_plus-00000-of-00001.parquet` is 129,408,418 bytes, and the ScanNet split's parquet is 12 KB.
- **Use here:** real, measured ground truth for the VLM-scoring step only. It is not an oracle source for the pipeline.

## 6. LOCO (tum-fml/loco)

- Repo: https://github.com/tum-fml/loco. Paper: https://mediatum.ub.tum.de/doc/1578845/1578845.pdf
- **License [V, `LICENSE`]:** CC0-1.0.
- **Content [V, survey]:** real logistics and warehouse images (37,988 images, 5,593 annotated), with COCO-format 2D boxes for pallet, small load carrier, stillage, forklift and pallet truck. No depth, no 3D.
- **Size [V, HEAD]:** one archive, `dataset.zip`, 769,055,500 bytes, covering all subsets (no per-split download), via https://go.mytum.de/239870 → an anonymous TUM webdisk download.
- **Use here:** measure the open-vocabulary detector's per-class recall and precision on real warehouse imagery. That is the perception stage where the vocabulary gap ("pallet" vs LVIS "palette", no rack class) shows up.

## 7. MMPTRACK — reviewed, not used

- Page: https://iccv2021-mmp.github.io/subpage/dataset.html; paper arXiv:2111.15157 (WACV 2023).
- **Content [V, page]:** multi-camera calibrated video (640×320, 15 FPS) of five staged environments built "in Microsoft indoor labs" (retail, lobby, industry, cafe, office). People only: 2D boxes with IDs, plus world-frame footpoints discretized to a voxel grid with units not stated. No depth.
- **Terms [V, the user agreement .docx]:**
  - Access requires emailing a signed agreement.
  - Use is limited to "non-commercial, research, or academic purposes" for "multi-camera multiple people tracking tasks".
  - Incorporating the dataset into "any other program, dataset, or product" is prohibited.
  - Derived techniques "can only be used for academic purposes".
- **Decision (user, 2026-10-26):** not used. The intended use (spatial QA, VLM scoring) is a grey area under the purpose clause, and a reply to a 2021 challenge inbox is unlikely within the deadline. It is still useful as evidence that real fixed-camera metric GT is scarce and research-only.

## 8. Other candidates (listed, not used)

| Dataset | License | What it has | Why not used now |
|---|---|---|---|
| Argoverse 2 Sensor | CC BY-NC-SA 4.0 [V] | Real driving, 30-class cuboids, lidar; 1 val log = 1.141 GB, public S3 [V] | A whole outdoor pipeline; 6-month candidate |
| nuScenes v1.0-mini | CC BY-NC-SA 4.0 + terms (account) [V] | Real driving, oriented boxes, lidar; 4.17 GB [V] | Same as above |
| CA-1M / CA-VQA (Apple) | CC BY-NC-ND 4.0 [V] | Real handheld homes, laser-registered GT, exhaustive boxes; CA-VQA is auto-generated spatial QA [V] | No-derivatives license; val is 63.9 GB. CA-VQA is the academic auto-QA to audit in the 6-month plan |
| Aria Digital Twin | Non-commercial; no derived works [V] | Egocentric real recordings; depth and segmentation rendered from a digital twin [V] | Sign-up; internal evaluation only |
| 4D-OR (hospital) | Access form, terms unseen [U] | 6 ceiling RGB-D Kinects, auto-annotated 3D object boxes [V] | Only real fixed multi-camera set with 3D object boxes found; access unknown |
| LOCO (tum-fml/loco) | CC0-1.0 [V] | Real logistics images, 2D boxes: pallet, small load carrier, stillage, forklift, pallet truck [V] | No depth or 3D; useful only to sanity-check detector prompts on real warehouses |
| Hypersim (apple/ml-hypersim) | CC BY-SA 3.0 [V] | Synthetic indoor, oriented boxes, depth (distance from camera centre) [V] | Not industrial; ShareAlike |
| Omni3D | CC BY-NC 4.0 [V] | Annotations only, unified camera-frame boxes [V] | Cite |
| ScanNet++ | Terms of use by application [V] | Laser scans, DSLR, iPhone RGB-D, instance labels; no 3D boxes [V] | Access delay; not industrial |
| MVTec ITODD | CC BY-NC-SA 4.0 [V] | Industrial objects, 6D pose [U] | Object pose, not scene QA |
| Voxel51 lidar-warehouse | CC-BY-SA-4.0 [V] | Real warehouse LiDAR, 3D boxes, 3.41 GB [V] | No RGB |
| BMW SORDI | Not stated [U] | Synthetic industrial, 2D [U] | No metric 3D |
| NVIDIA SimReady-Warehouse-01 | CC-BY-4.0 [V] | USD assets (racks, pallets) | Would require rendering our own scenes |

## Corrections to the original brief

1. ARKitScenes is no longer CC BY-NC-SA 4.0 (see §2) [V].
2. Smart Spaces "depth in 2025/2026 parts" is only partly true: 2026 test scenes have no depth, and 2025 test scenes have no GT [V].
3. Smart Spaces 2026 also includes 2 real test scenes and 6 Cosmos-Transfer re-renders [V].
4. Smart Spaces 2026 measures 2.478 TB, not the card's 2.94 TB [V].
5. arXiv:2610.02580 is correct for Smart Spaces, but it is a tracking benchmark; the warehouse spatial QA data is a separate dataset (§4) [V].
