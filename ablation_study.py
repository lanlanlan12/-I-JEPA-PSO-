import os
import importlib.util
import torch

def _load_main_module():
    path = os.path.join(os.path.dirname(__file__), "1.py")
    spec = importlib.util.spec_from_file_location("main_1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    mod = _load_main_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("dataset_summary.csv"):
        print("Warning: dataset_summary.csv not found. Please provide data.")
        return
    dataset = mod.LongJumpDataset("dataset_summary.csv", max_len=320, analyze_kinematics=False, phase="all")
    mod.run_ablation_study(dataset, device)

if __name__ == "__main__":
    main()
