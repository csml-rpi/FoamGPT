import os
import argparse
import numpy as np
import pandas as pd
import pyvista as pv

from benchmark_utils import DIR_ADVANCED, EXP_ADVANCED, MODEL



def success(case_path: str) -> bool:
    if not os.path.exists(case_path):
        return 0

    llm_run_folders = [f for f in os.listdir(case_path)
                        if os.path.isdir(os.path.join(case_path, f))]
    if not llm_run_folders:
        return 0

    success = 0

    log_files = [f for f in os.listdir(case_path) if f.startswith("log.") and f.endswith("Foam")]
    for log_file in log_files:
        log_path = os.path.join(case_path, log_file)
        try:
            with open(log_path, "r") as file:
                lines = file.readlines()
                last_line = lines[-2].strip() if len(lines) > 1 else ""
                if last_line == "End":
                    success = 1
                # else:
                #     print(case_path)
        except:
            success = 0
            break
    return success



def touch_foam_file(directory, file_name):
    foam_path = os.path.join(directory, file_name)
    with open(foam_path, 'w') as f:
        pass
    return foam_path

def calculate_nmse(array1, array2):
    return np.sum((array1 - array2) ** 2) / np.sum(array1 ** 2)

def align_and_scale_mesh(gt_mesh, meta_mesh):
    gt_bounds = np.array(gt_mesh[0].bounds).reshape(3, 2)
    meta_bounds = np.array(meta_mesh[0].bounds).reshape(3, 2)

    if np.allclose(gt_bounds, meta_bounds, atol=1e-6):
        return meta_mesh

    shift = gt_bounds[:, 0] - meta_bounds[:, 0]
    gt_size = gt_bounds[:, 1] - gt_bounds[:, 0]
    meta_size = meta_bounds[:, 1] - meta_bounds[:, 0]
    scaling = np.divide(gt_size, meta_size, out=np.ones_like(gt_size), where=meta_size != 0)

    aligned_meta_mesh = meta_mesh[0].copy()
    aligned_meta_mesh.points = (aligned_meta_mesh.points - meta_bounds[:, 0]) * scaling + gt_bounds[:, 0]
    return [aligned_meta_mesh]

def get_inner_run_folder(path):
    for f in os.listdir(path):
        full = os.path.join(path, f)
        if os.path.isdir(full) and f != "GT_Files":
            return full
    return None

def evaluate_nmse(DATASETS_ADVANCED, EXP_ADVANCED):
    fields = ["U", "p", "rho", "T"]
    gt_foam = touch_foam_file(DATASETS_ADVANCED, os.path.basename(DATASETS_ADVANCED) + ".foam")
    llm_foam = touch_foam_file(EXP_ADVANCED, os.path.basename(EXP_ADVANCED) + ".foam")

    try:
        gt_reader = pv.OpenFOAMReader(gt_foam)
        last_time = gt_reader.time_values[-1]
        gt_reader.set_active_time_value(last_time)
        meta_reader = pv.OpenFOAMReader(llm_foam)
        closest_time = min(meta_reader.time_values, key=lambda x: abs(x - last_time))
        meta_reader.set_active_time_value(closest_time)
    except:
        return 9999

    gt_mesh = gt_reader.read()
    meta_mesh = meta_reader.read()
    meta_mesh = align_and_scale_mesh(gt_mesh, meta_mesh)

    scores = []
    for field in fields:
        if field in gt_mesh[0].cell_data and field in meta_mesh[0].cell_data:
            try:
                if gt_mesh[0].n_cells != meta_mesh[0].n_cells:
                    try:
                        OG_mesh = gt_mesh.copy()
                        if field in OG_mesh[0].point_data:
                            del OG_mesh[0].point_data[field]
                        if field in OG_mesh[0].cell_data:
                            del OG_mesh[0].cell_data[field]
                        projected_mesh = [OG_mesh[0].interpolate(meta_mesh[0], sharpness=0.1, n_points=50)]
                        projected_mesh[0] = projected_mesh[0].point_data_to_cell_data()
                        meta_data = projected_mesh[0][field]
                    except:
                        scores.append(9999)
                        continue
                else:
                    meta_data = meta_mesh[0][field]

                gt_data = gt_mesh[0][field]
                score = calculate_nmse(gt_data, meta_data)
                scores.append(score)
            except:
                scores.append(9999)
    if scores:
        return np.mean(scores)
    else:
        return 9999

def process_all(model=MODEL, exp_advanced=EXP_ADVANCED, llm_runs_dir=None, base_path=DIR_ADVANCED):
    records = []
    datasets = sorted(os.listdir(base_path))
    llm_base_path = llm_runs_dir or os.path.join(exp_advanced, model, "runs")

    for dataset in datasets:
        dataset_dir = os.path.join(base_path, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        gt_dir = os.path.join(dataset_dir, "GT_Files")
        llm_dir = os.path.join(llm_base_path, dataset)
        if not gt_dir or not llm_dir or not os.path.exists(gt_dir) or not os.path.exists(llm_dir):
            print(f"{llm_dir} error")
            score = 9999
        elif not success(llm_dir):
            score = 9999
        else:
            score = evaluate_nmse(gt_dir, llm_dir)
        records.append([dataset, score])
    return pd.DataFrame(records, columns=["Dataset", "NMSE"])

# Helper: NMSE threshold scoring
def score_nmse(val):
    if val < 0.1:
        return 1.0
    elif val < 0.3:
        return 0.5
    else:
        return 0.0

def evaluate(nmse_df):
    nmse_df["NMSE_Score"] = nmse_df["NMSE"].apply(score_nmse)

    return pd.DataFrame({
        "NMSE Score": [nmse_df["NMSE_Score"].mean()],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate advanced-case NMSE for benchmark run folders.")
    parser.add_argument("--model", default=MODEL, help="Model label under the experiment root.")
    parser.add_argument("--exp-root", default=EXP_ADVANCED, help="Experiment root containing <model>/runs.")
    parser.add_argument("--llm-runs-dir", default=None, help="Direct path to a runs directory; overrides --exp-root/--model.")
    parser.add_argument("--dataset-root", default=DIR_ADVANCED, help="Advanced dataset root containing GT_Files.")
    parser.add_argument("--output-prefix", default=None, help="Prefix for output CSV files; defaults to --model.")
    args = parser.parse_args()

    report_prefix = args.output_prefix or args.model

    basic_df = process_all(
        model=args.model,
        exp_advanced=args.exp_root,
        llm_runs_dir=args.llm_runs_dir,
        base_path=args.dataset_root,
    )
    csv = f"{report_prefix}-mse-report-advanced.csv"
    basic_df.to_csv(csv, index=False)

    nmse = pd.read_csv(csv)
    basic_scores = evaluate(nmse)

    basic_scores.to_csv(f"{report_prefix}-final-score-advanced.csv", index=False)
