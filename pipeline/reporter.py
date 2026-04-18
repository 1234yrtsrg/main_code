import numpy as np
import pandas as pd
from utils.logger import Logger
from utils.metrics import MetricsUtil

class ResultReporter:
    @staticmethod
    def print_shared_bands(shared_idx, wavelength_cols):
        shared_waves = wavelength_cols[shared_idx]
        shared_waves_float = pd.to_numeric(pd.Index(shared_waves), errors="coerce").values
        
        order_by_wave = np.argsort(shared_waves_float)
        Logger.log("[SharedBands] ===== Print in ascending order by wavelength =====")
        for k, oi in enumerate(order_by_wave, start=1):
            Logger.log(f"[SharedBands] #{k:02d} idx={shared_idx[oi]:4d} wave={shared_waves_float[oi]:.2f} nm (col='{shared_waves[oi]}')")

        return shared_waves, shared_waves_float

    @staticmethod
    def export_oof_predictions_to_excel(output_path, sample_ids, outer_fold_ids, target_metals, Y_true_matrix, oof_pred_global_dict, oof_pred_moe_dict, fold_metrics_df):
        sample_ids = np.asarray(sample_ids)
        outer_fold_ids = np.asarray(outer_fold_ids)

        df_g = pd.DataFrame({"Sample": sample_ids, "OuterFold": outer_fold_ids.astype(int)})
        df_m = pd.DataFrame({"Sample": sample_ids, "OuterFold": outer_fold_ids.astype(int)})

        for j, metal in enumerate(target_metals):
            y_true = np.asarray(Y_true_matrix[:, j], dtype=float)
            pred_g = np.asarray(oof_pred_global_dict[metal], dtype=float)
            pred_m = np.asarray(oof_pred_moe_dict[metal], dtype=float)

            df_g[f"{metal}_true"] = y_true
            df_g[f"{metal}_pred"] = pred_g
            df_m[f"{metal}_true"] = y_true
            df_m[f"{metal}_pred"] = pred_m

        mean_metrics = fold_metrics_df.groupby(['Target', 'Model', 'Set'])[['R2', 'RMSE', 'RPD']].mean().reset_index()

        pivot_df = mean_metrics.pivot(index=['Target', 'Model'], columns='Set', values=['R2', 'RMSE', 'RPD'])

        pivot_df.columns = [f"{col[1]} {col[0]}" for col in pivot_df.columns]
        pivot_df = pivot_df.reset_index()

        if "Test R2" in pivot_df.columns:
            pivot_df = pivot_df.rename(columns={
                "Test R2": "OOF R2",
                "Test RMSE": "OOF RMSE",
                "Test RPD": "OOF RPD"
            })

        cols_order = ['Target', 'Model', 'Train R2', 'Train RMSE', 'Train RPD', 'OOF R2', 'OOF RMSE', 'OOF RPD']
        valid_cols = [c for c in cols_order if c in pivot_df.columns]
        df_summary = pivot_df[valid_cols]

        model_order = {"PLS": 1, "KRR": 2, "ElasticNet": 3, "RFRR": 4, "RSRidge": 5, "GlobalStack": 6, "MoE": 7}
        df_summary = df_summary.copy()
        df_summary['_sort_idx'] = df_summary['Model'].map(model_order).fillna(99)
        df_summary = df_summary.sort_values(by=['Target', '_sort_idx']).drop(columns=['_sort_idx'])

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_g.to_excel(writer, sheet_name="GlobalStack_Predictions", index=False)
            df_m.to_excel(writer, sheet_name="MoE_Predictions", index=False)
            fold_metrics_df.to_excel(writer, sheet_name="Fold_Performance", index=False)
            df_summary.to_excel(writer, sheet_name="Global_Summary", index=False)

        Logger.log(f"[SAVE] OOF predictions and summaries saved to: {output_path}")