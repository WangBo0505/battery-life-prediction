import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import streamlit as st
import time

# ===================== 全局配置 =====================
st.set_page_config(
    page_title="储能电池全生命周期预测系统",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

LOGO_URL = "https://www.ptl-global.com/cn/img/logo.png"

st.markdown(f"""
    <style>
        .fixed-logo {{
            position: fixed;
            top: 100px;
            right: 30px;
            width: 120px;
            z-index: 9999;
        }}
    </style>
    <img src="{LOGO_URL}" class="fixed-logo" alt="logo">
""", unsafe_allow_html=True)

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['text.color'] = '#2c3e50'
plt.rcParams['axes.labelcolor'] = '#2980b9'
plt.rcParams['xtick.color'] = '#34495e'
plt.rcParams['ytick.color'] = '#34495e'
plt.rcParams['axes.edgecolor'] = '#bdc3c7'
plt.rcParams['grid.color'] = '#ecf0f1'
plt.rcParams['grid.alpha'] = 0.8

# ======================================
# 模型代码（全部保留，不改动）
# ======================================
R_GAS = 8.314462618

@dataclass
class ColumnMap:
    cycle: str = "cycle"
    cap_ah: str = "capacity_ah"
    temp_c: str = "temp_c_mean"
    dod: str = "dod"
    i_dis_a: str = "i_dis_a_mean"
    rated_cap_ah: Optional[str] = None

@dataclass
class FitConfig:
    soh_target: float = 0.80
    min_dod: float = 0.50
    min_cycles: int = 20
    use_efc: bool = True
    bootstrap_n: int = 50
    random_seed: int = 42
    temp_min_c: float = 20.0
    temp_max_c: float = 40.0
    smooth_method: str = "median"
    smooth_window: int = 5
    outlier_threshold: float = 3.0

def preprocess_smooth_data(df: pd.DataFrame, cmap: ColumnMap, cfg: FitConfig) -> pd.DataFrame:
    d = df.copy()
    d = d.dropna(subset=[cmap.cap_ah, cmap.cycle])
    d = d[d[cmap.cap_ah].astype(float) >= 0]
    d = d.sort_values(cmap.cycle).reset_index(drop=True)
    cap_series = d[cmap.cap_ah].astype(float)
    cap_mean = cap_series.mean()
    cap_std = cap_series.std()
    if cap_std == 0:
        cap_clean = cap_series
        outliers = pd.Series([False]*len(cap_series))
    else:
        lower_bound = cap_mean - cfg.outlier_threshold * cap_std
        upper_bound = cap_mean + cfg.outlier_threshold * cap_std
        cap_clean = cap_series.copy()
        outliers = (cap_clean < lower_bound) | (cap_clean > upper_bound)
        cap_clean = cap_clean.mask(outliers, np.nan).interpolate().bfill().ffill()
    cap_array = cap_clean.values
    window_size = cfg.smooth_window
    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, len(cap_array))
    if window_size < 3:
        window_size = 3 if len(cap_array) >=3 else len(cap_array)
    if cfg.smooth_method == "savgol" and len(cap_array) >= window_size:
        cap_smoothed = savgol_filter(cap_array, window_size, polyorder=1)
    elif cfg.smooth_method == "median":
        cap_smoothed = median_filter(cap_array, size=window_size)
    elif cfg.smooth_method == "rolling":
        cap_smoothed = pd.Series(cap_array).rolling(window=window_size, center=True, min_periods=1).mean().values
    else:
        cap_smoothed = cap_array
    d[cmap.cap_ah] = cap_smoothed
    if cmap.temp_c in d.columns:
        temp_series = d[cmap.temp_c].astype(float).interpolate().bfill().ffill()
        d[cmap.temp_c] = temp_series.rolling(window=3, center=True, min_periods=1).mean().values
    if cmap.dod in d.columns:
        d[cmap.dod] = d[cmap.dod].astype(float).fillna(1.0)
        d[cmap.dod] = np.clip(d[cmap.dod], 0.1, 1.0)
    return d

def compute_features(df: pd.DataFrame, cmap: ColumnMap, rated_capacity_input: Optional[float] = None) -> Tuple[pd.DataFrame, float, float]:
    d = df.copy()
    required_cols = [cmap.cycle, cmap.cap_ah, cmap.temp_c, cmap.dod, cmap.i_dis_a]
    missing_cols = [col for col in required_cols if col not in d.columns]
    if missing_cols:
        raise ValueError(f"CSV文件缺少必要列: {missing_cols}")
    d = d.sort_values(cmap.cycle).drop_duplicates(subset=[cmap.cycle]).reset_index(drop=True)
    cap_series = d[cmap.cap_ah].astype(float)
    if cmap.rated_cap_ah and cmap.rated_cap_ah in d.columns:
        rated_cap_csv = d[cmap.rated_cap_ah].astype(float)
        Q0 = float(rated_cap_csv.iloc[0])
    else:
        Q0 = float(cap_series.head(20).median())
    Rated_Cap = rated_capacity_input if (rated_capacity_input and rated_capacity_input > 0) else Q0
    cap = cap_series.to_numpy()
    temp_c = d[cmap.temp_c].astype(float).to_numpy()
    dod = d[cmap.dod].astype(float).to_numpy()
    i_dis = d[cmap.i_dis_a].astype(float).to_numpy()
    soh_calc = cap / Q0
    dQ = np.clip(1.0 - soh_calc, 1e-6, 0.4)
    soh_show = cap / Rated_Cap
    c_rate = np.clip(np.abs(i_dis) / max(Q0, 1e-6), 1e-6, None)
    efc = np.cumsum(np.clip(dod, 0.0, 1.0))
    d["Q0_ah"] = Q0
    d["Rated_Cap_Ah"] = Rated_Cap
    d["soh"] = soh_show
    d["dQ"] = dQ
    d["c_rate"] = c_rate
    d["efc"] = efc
    d["temp_k"] = temp_c + 273.15
    return d, Q0, Rated_Cap

def _model_log_dQ(params, N, dod, c_rate):
    logk, alpha, beta, gamma = params
    N = np.clip(N, 1e-6, None)
    dod = np.clip(dod, 1e-6, None)
    c_rate = np.clip(c_rate, 1e-6, None)
    return (logk + alpha * np.log(N) + beta * np.log(dod) + gamma * np.log(c_rate))

def fit_life_model(df_feat: pd.DataFrame, cmap: ColumnMap, cfg: FitConfig):
    d = df_feat.copy()
    d = d[(d["soh"] > 0.5) & (d["soh"] < 1.1)]
    d = d[(d[cmap.temp_c] >= cfg.temp_min_c) & (d[cmap.temp_c] <= cfg.temp_max_c)]
    d = d[d[cmap.dod].astype(float) >= cfg.min_dod]
    if len(d) < cfg.min_cycles:
        d = df_feat.copy()
        d = d[(d[cmap.temp_c] >= cfg.temp_min_c - 10) & (d[cmap.temp_c] <= cfg.temp_max_c + 10)]
        d = d[d[cmap.dod].astype(float) >= 0.1]
    if len(d) < 5:
        raise ValueError(f"数据量过少：仅{len(d)}个有效循环，无法进行拟合")
    N = d["efc"].to_numpy() if cfg.use_efc else d[cmap.cycle].astype(float).to_numpy()
    dod = d[cmap.dod].astype(float).to_numpy()
    c_rate = d["c_rate"].to_numpy()
    y = np.log(d["dQ"].to_numpy())
    x0 = np.array([-7.0, 0.8, 0.5, 0.1], dtype=float)
    lb = np.array([-20.0, 0.2, 0.1, 0.0], dtype=float)
    ub = np.array([-3.0, 2.5, 3.0, 2.0], dtype=float)
    def residuals(p):
        return _model_log_dQ(p, N, dod, c_rate) - y
    res = least_squares(residuals, x0=x0, bounds=(lb, ub), loss="soft_l1", f_scale=1.0, max_nfev=5000, gtol=1e-4)
    p_hat = res.x
    rmse_log = float(np.sqrt(np.mean(res.fun ** 2)))
    y_pred = _model_log_dQ(p_hat, N, dod, c_rate)
    dQ_pred = np.exp(y_pred)
    params_dict = {
        "k": float(np.exp(p_hat[0])),
        "logk": float(p_hat[0]),
        "alpha": float(p_hat[1]),
        "beta_dod": float(p_hat[2]),
        "gamma_crate": float(p_hat[3])
    }
    out = {
        "params": params_dict,
        "rmse_log": rmse_log,
        "n_used": int(len(d)),
        "use_efc": cfg.use_efc,
        "y_true": y,
        "y_pred": y_pred,
        "dQ_pred": dQ_pred,
        "filtered_df": d,
        "fit_params": p_hat
    }
    return out

def solve_life_to_target(params: Dict[str, float], target_soh: float, dod_ref: float, c_rate_ref: float) -> float:
    dQ_target = np.clip(1.0 - target_soh, 1e-6, 0.4)
    k = params["k"]
    alpha = params["alpha"]
    beta = params["beta_dod"]
    gamma = params["gamma_crate"]
    denom = k * (np.clip(dod_ref, 1e-6, None) ** beta) * (np.clip(c_rate_ref, 1e-6, None) ** gamma)
    denom = max(denom, 1e-30)
    N = (dQ_target / denom) ** (1.0 / max(alpha, 1e-6))
    return float(N)

def linear_fit_decay(df: pd.DataFrame, soh_target: float = 0.80) -> Dict:
    cycle = df["cycle"].values
    cap = df["capacity_ah"].values
    Q0 = cap[0] if cap[0] > 0 else cap.mean()
    soh = cap / Q0
    k, b = np.polyfit(cycle, soh, deg=1)
    cycle_end = (soh_target - b) / k
    cycle_end = max(cycle_end, 1)
    cycle_ext = np.arange(1, int(cycle_end) + 1)
    soh_ext = k * cycle_ext + b
    soh_ext = np.clip(soh_ext, 0.78, 1.05)
    pred_df = pd.DataFrame({
        "Pred_EFC": cycle_ext.astype(int),
        "Pred_SOH": soh_ext,
        "Pred_Capacity(Ah)": soh_ext * Q0,
        "Capacity_Decay": 1 - soh_ext
    })
    result = {
        "Q0": Q0,
        "decay_coeff": k,
        "intercept": b,
        "soh_target": soh_target,
        "life_cycles": cycle_end,
        "cycle_measured": cycle,
        "soh_measured": soh,
        "cycle_extended": cycle_ext,
        "soh_extended": soh_ext,
        "predict_full_df": pred_df
    }
    return result

def run_pipeline(csv_file, cmap: ColumnMap, cfg: FitConfig, ref_conditions: Dict[str, float], rated_capacity_input: Optional[float]):
    df = pd.read_csv(csv_file)
    df_smoothed = preprocess_smooth_data(df, cmap, cfg)
    df_feat, Q0_semi, Rated_Cap = compute_features(df_smoothed, cmap, rated_capacity_input)
    fit_semi = fit_life_model(df_feat, cmap, cfg)
    Nlife_semi = solve_life_to_target(fit_semi["params"], cfg.soh_target, ref_conditions["dod"], ref_conditions["c_rate"])
    linear_result = linear_fit_decay(df, cfg.soh_target)
    result = {
        "semi_Q0": Q0_semi,
        "semi_fit": fit_semi,
        "semi_life_N": Nlife_semi,
        "semi_feat_df": df_feat,
        "linear_result": linear_result,
        "Rated_Cap_Ah": Rated_Cap,
        "ref_conditions": ref_conditions,
        "cfg": cfg
    }
    return result

# ======================================
# 主界面（动态绘图已修改为每秒200圈）
# ======================================
def main():
    st.markdown("""
        <h1 style='text-align: center; color: #2980b9; font-weight: bold;'>🔋 储能电池全生命周期高精度预测系统</h1>
        <h3 style='text-align: center; color: #7f8c8d;'>Electrochemical Attenuation Model | Full Cycle Capacity Prediction</h3>
        <hr style='border: 1px solid #ecf0f1;'>
    """, unsafe_allow_html=True)

    cmap = ColumnMap()
    col1, col2 = st.columns([1, 2.8], gap="large")

    with col1:
        st.markdown("<h4 style='color: #2980b9; border-bottom:2px solid #3498db; padding-bottom:5px'>⚙️ 参数配置</h4>", unsafe_allow_html=True)
        rated_capacity = st.number_input("额定容量 (Rated Capacity) (Ah)", min_value=0.01, max_value=10000.0, value=None, step=0.01, format="%.2f")
        target_soh = st.number_input("寿命终点SOH值 (Target SOH)", min_value=0.6, max_value=0.95, value=0.80, step=0.01, format="%.2f")
        dod_ref = st.number_input("放电深度 (Depth of Discharge)", min_value=0.0, max_value=1.0, value=1.0, step=0.01, format="%.2f")
        c_rate_ref = st.number_input("放电倍率 (C-rate)", min_value=0.01, max_value=5.0, value=0.5, step=0.01, format="%.2f")

        st.markdown("<h4 style='color: #2980b9; border-bottom:2px solid #3498db; padding-bottom:5px; margin-top:20px'>📂 上传数据</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("选择CSV文件 (Select CSV File)", type="csv")
        run_btn = st.button("开始预测 (Start Prediction)", use_container_width=True, type="primary")

    with col2:
        if run_btn and uploaded_file is not None:
            try:
                with st.spinner("Calculating..."):
                    cfg = FitConfig(soh_target=target_soh)
                    ref_conditions = {"temp_c":25.0, "dod":dod_ref, "c_rate":c_rate_ref}
                    all_result = run_pipeline(uploaded_file, cmap, cfg, ref_conditions, rated_capacity)
                    linear_res = all_result["linear_result"]
                    life_cycle = int(linear_res["life_cycles"])
                    decay_coeff = linear_res["decay_coeff"]
                    Q0_linear = linear_res["Q0"]
                    rated_cap = all_result["Rated_Cap_Ah"]
                    pred_df = linear_res["predict_full_df"]
                    feat_df = all_result["semi_feat_df"]

                st.markdown("<h4 style='color: #2980b9; border-bottom:2px solid #3498db; padding-bottom:5px'>📊 预测结果</h4>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(f"""
                    <div style='color: #2c3e50; font-size: 14px; line-height: 1.8;'>
                    初始容量 Q0: {Q0_linear:.3f} Ah | 额定容量: {rated_cap:.3f} Ah<br>
                    衰减系数: {decay_coeff:.8f} SOH/圈<br>
                    每圈SOH下降: {abs(decay_coeff):.8f}<br>
                    目标SOH: {target_soh*100:.1f}% | 放电深度: {dod_ref*100:.1f}% | 放电倍率: {c_rate_ref}C<br>
                    预测总循环数: <span style='color: #e67e22; font-weight: bold; font-size:15px;'>{life_cycle}</span> 圈
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<h4 style='color: #2980b9; border-bottom:2px solid #3498db; padding-bottom:5px; marginTop:10px'>📈 SOH 衰减曲线</h4>", unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(12, 5), dpi=100)

                # 实测点
                ax.scatter(linear_res["cycle_measured"], linear_res["soh_measured"], 
                           c='#2980b9', s=18, label='Actual SOH', alpha=0.85)
                ax.axhline(y=target_soh, color='#e74c3c', linestyle='--', linewidth=2, 
                           label=f'End of Life ({target_soh*100:.1f}% SOH)')
                ax.set_xlabel("Cycle")
                ax.set_ylabel("SOH")
                ax.set_ylim(0.58, 1.05)
                ax.grid(True, alpha=0.4)
                ax.legend(loc='upper right')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                plot_placeholder = st.empty()
                plot_placeholder.pyplot(fig)

                # ======================
                # 关键：每秒画 200 圈
                # ======================
                cycle_ext = linear_res["cycle_extended"]
                soh_ext = linear_res["soh_extended"]

                cycles_per_second = 200
                total_cycles = len(cycle_ext)
                draw_steps = max(10, total_cycles // 50)  # 平滑度

                for i in range(0, total_cycles, draw_steps):
                    ax.plot(cycle_ext[:i+draw_steps], soh_ext[:i+draw_steps], 
                            color='#e67e22', linewidth=2.5, label='Prediction' if i == 0 else "")
                    plot_placeholder.pyplot(fig)
                    time.sleep(draw_steps / cycles_per_second)

                # 最终寿命线
                ax.axvline(x=life_cycle, color='#27ae60', linestyle=':', linewidth=2, 
                           label=f'Predicted Life: {life_cycle} cycles')
                ax.legend(loc='upper right')
                plot_placeholder.pyplot(fig)

                # 导出
                st.markdown("<h4 style='color: #2980b9; border-bottom:2px solid #3498db; padding-bottom:5px; margin-top:10px'>💾 数据导出</h4>", unsafe_allow_html=True)
                real_data = pd.DataFrame({
                    "Cycle": linear_res["cycle_measured"],
                    "Capacity(Ah)": linear_res["soh_measured"] * Q0_linear,
                    "SOH": linear_res["soh_measured"]
                })
                export_df = pd.concat([real_data, pred_df], ignore_index=True)
                csv_data = export_df.to_csv(index=False, encoding="utf-8-sig").encode('utf-8-sig')
                st.download_button("Download Prediction Data", csv_data, 
                                   "Battery_Life_Prediction.csv", "text/csv", use_container_width=True)

            except Exception as e:
                st.error(f"计算失败：{str(e)}")

        elif run_btn:
            st.warning("请先上传CSV文件")
        else:
            st.info("👈 上传CSV文件并点击开始预测")

if __name__ == "__main__":
    main()
