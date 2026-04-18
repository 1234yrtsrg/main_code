import os
import warnings
from sklearn.exceptions import ConvergenceWarning
from pipeline.orchestrator import NestedCVOrchestrator
from tabpfn import TabPFNRegressor
import numpy as np

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['XDG_CACHE_HOME'] = '/storage/v-jinpewang/az_workspace/zhanglin/reproduction/specml/tabpfn_cache'
print("正在预先下载并缓存 TabPFN 模型...")
dummy_model = TabPFNRegressor()
dummy_model.fit(np.array([[0.0]]), np.array([0.0]))
print("模型下载与缓存完成！")


def suppress_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="^Objective did not converge.*")
    warnings.filterwarnings("ignore", category=UserWarning, message="Using a target size .*different to the input size.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in cast")
    warnings.filterwarnings("ignore", message=".*A worker stopped while some jobs were given to the executor.*")

if __name__ == "__main__":
    suppress_warnings()
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "data/Raman_spectroscopy_data_preprocessed.csv"))
    OUTPUT_XLSX_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "nestedcv_oof_predictions.xlsx"))
    TARGET_METALS = ["118Sn (KED)", "209Bi (KED)"]

    
    print(f"[*] Data Source: {DATA_PATH}")
    print(f"[*] Output Target: {OUTPUT_XLSX_PATH}")

    runner = NestedCVOrchestrator(
        data_path=DATA_PATH,
        output_path=OUTPUT_XLSX_PATH,
        target_metals=TARGET_METALS
    )
    runner.run()