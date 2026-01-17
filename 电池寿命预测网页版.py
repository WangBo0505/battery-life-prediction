import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import streamlit as st

# ===================== 全局配置 - 科研风 适配云端 =====================
st.set_page_config(
    page_title="储能电池全生命周期预测系统",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
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
# ↓↓↓↓↓↓ 核心代码 + Q0优化适配 完整保留 ↓↓↓↓↓↓
# ======================================
R_GAS = 8.314462618  # 理想气体常数

@dataclass
class ColumnMap:
    cycle: str = "cycle"
    cap_ah: str = "capacity_ah"
    temp_c: str = "temp_c_mean"
    dod: str = "dod"
    i_dis_a: str = "i_dis_a_mean"

@dataclass
class FitConfig:
    soh_target: float = 0.80
    min_dod: float = 0.80
    min_cycles: int = 50
    use_efc: bool = True
    bootstrap_n: int = 100
    random_seed: int = 42
    temp_min_c: float = 20.0  # 过滤原始测试数据：仅拟合恒温区间数据
    temp_max_c: float = 55.0

def compute_features(df: pd.DataFrame, cmap: ColumnMap, manual_Q0: Optional[float] = None) -> Tuple[pd.DataFrame, float, str]:
    """
    计算特征值 + 适配Q0双模式
    manual_Q0: 用户手动输入的额定容量，None则自动取capacity_ah最大值
    return: 处理后数据, Q0值, Q0来源描述
    """
    d = df.copy()
    required_cols = [cmap.cycle, cmap.cap_ah, cmap.temp_c, cmap.dod, cmap.i_dis_a]
    missing_cols = [col for col in required_cols if col not in d.columns]
    if missing_cols:
        raise ValueError(f"CSV文件缺少必要列: {missing_cols}")

    d = d.sort_values(cmap.cycle).drop_duplicates(subset=[cmap.cycle]).reset_index(drop=True)
    cap_series = d[cmap.cap_ah].astype(float)
    
    # ✅ 核心优化：Q0取值逻辑 手动输入优先，否则取容量最大值
    if manual_Q0 is not None and manual_Q0 > 0:
        Q0 = round(float(manual_Q0), 3)
        q0_source = "手动输入(额定值)"
    else:
        Q0 = round(float(cap_series.max()), 3)  # 自动取实测容量最大值
        q0_source = "自动计算(最大值)"

    cap = cap_series.to_numpy()
    temp_c = d[cmap.temp_c].astype(float).to_numpy()
    dod = d[cmap.dod].astype(float).to_numpy()
    i_dis = d[cmap.i_dis_a].astype(float).to_numpy()

    soh = cap / Q0
    dQ = np.clip(1.0 - soh, 1e-6, 0.4)
    c_rate = np.clip(np.abs(i_dis) / max(Q0, 1e-6), 1e-6, None)
    efc = np.cumsum(np.clip(dod, 0.0, 1.0))

    d["Q0_ah"] = Q0
    d["soh"] = soh
    d["dQ"] = dQ
    d["c_rate"] = c_rate
    d["efc"] = efc
    d["temp_k"] = temp_c + 273.15

    return d, Q0, q0_source

def _model_log_dQ(params, N, dod, c_rate):
    logk, alpha, beta, gamma = params
    N = np.clip(N, 1e-6, None)
    dod = np.clip(dod, 1e-6, None)
    c_rate = np.clip(c_rate, 1e-6, None)
    return (logk + alpha * np.log(N) + beta * np.log(dod) + gamma * np.log(c_rate))

def fit_life_model(df_feat: pd.DataFrame, cmap: ColumnMap, cfg: FitConfig):
    d = df_feat.copy()
    d = d[(d["soh"] > 0.6) & (d["soh"] < 0.98)]
    d = d[(d[cmap.temp_c] >= cfg.temp_min_c) & (d[cmap.temp_c] <= cfg.temp_max_c)]
    d = d[d[cmap.dod].astype(float) >= cfg.min_dod]

    if len(d) < cfg.min_cycles:
        raise ValueError(f"数据量不足：有效循环 {len(d)} < min_cycles={cfg.min_cycles}")

    N = d["efc"].to_numpy() if cfg.use_efc else d[cmap.cycle].astype(float).to_numpy()
    dod = d[cmap.dod].astype(float).to_numpy()
    c_rate = d["c_rate"].to_numpy()
    y = np.log(d["dQ"].to_numpy())

    x0 = np.array([-8.0, 1.0, 0.8, 0.2], dtype=float)
    lb = np.array([-15.0, 0.6, 0.2, 0.0], dtype=float)
    ub = np.array([-5.0, 1.8, 2.0, 1.5], dtype=float)

    def residuals(p):
        return _model_log_dQ(p, N, dod, c_rate) - y

    res = least_squares(residuals, x0=x0, bounds=(lb, ub), loss="huber",
                        f_scale=0.5, max_nfev=8000, gtol=1e-5)

    p_hat = res.x
    rmse_log = float(np.sqrt(np.mean(res.fun ** 2)))
    y_pred = _model_log_dQ(p_hat, N, dod, c_rate)
    dQ_pred = np.exp(y_pred)

    params_dict = {
        "k": float(np.exp(p_hat[0])),
        "logk": float(p_hat[0]),
        "alpha": float(p_hat[1]),
        "beta_dod": float(p_hat[2]),
        "gamma_crate": float(p_hat[3]),
    }

    out = {
        "params": params_dict,
        "rmse_log_dQ": rmse_log,
        "n_used": int(len(d)),
        "use_efc": cfg.use_efc,
        "y_true": y,
        "y_pred": y_pred,
        "dQ_pred": dQ_pred,
        "filtered_df": d,
        "fit_params": p_hat
    }
    return out

def solve_life_to_target(params: Dict[str, float],
                         target_soh: float,
                         dod_ref: float,
                         c_rate_ref: float) -> float:
    dQ_target = np.clip(1.0 - target_soh, 1e-6, 0.4)
    k = params["k"]
    alpha = params["alpha"]
    beta = params["beta_dod"]
    gamma = params["gamma_crate"]
    denom = k * (np.clip(dod_ref, 1e-6, None) ** beta) * (np.clip(c_rate_ref, 1e-6, None) ** gamma)
    denom = max(denom, 1e-30)
    N = (dQ_target / denom) ** (1.0 / max(alpha, 1e-6))
    return float(N)

def bootstrap_life_ci(df_feat: pd.DataFrame, cmap: ColumnMap, cfg: FitConfig,
                      dod_ref: float, c_rate_ref: float) -> Tuple[float, float]:
    rng = np.random.default_rng(cfg.random_seed)
    d = df_feat.copy()
    d = d[(d["soh"] > 0.6) & (d["soh"] < 0.98)]
    d = d[(d[cmap.temp_c] >= cfg.temp_min_c) & (d[cmap.temp_c] <= cfg.temp_max_c)]
    d = d[d[cmap.dod].astype(float) >= cfg.min_dod].reset_index(drop=True)

    life_samples = []
    n = len(d)
    fail_count = 0
    max_fail = cfg.bootstrap_n // 3

    for _ in range(cfg.bootstrap_n):
        if fail_count > max_fail:
            break
        try:
            idx = rng.integers(0, n, size=n)
            sample = d.iloc[idx].sort_values(cmap.cycle).reset_index(drop=True)
            fit = fit_life_model(sample, cmap, cfg)
            Nlife = solve_life_to_target(fit["params"], cfg.soh_target, dod_ref, c_rate_ref)
            if np.isfinite(Nlife) and 100 < Nlife < 5000:
                life_samples.append(Nlife)
        except:
            fail_count += 1
            continue

    if len(life_samples) < max(20, cfg.bootstrap_n * 0.2):
        raise RuntimeError("bootstrap有效样本过少")

    lo, hi = np.percentile(life_samples, [2.5, 97.5])
    return float(lo), float(hi)

# ======================================
# ✅ 全循环容量预测函数 - 适配所有工况参数
# ======================================
def predict_full_life_cycles(fit_result, Q0, target_soh, life_cycles, dod_ref=1.0, c_rate_ref=0.5):
    logk, alpha, beta, gamma = fit_result["fit_params"]
    pred_efc = np.linspace(1, life_cycles, int(life_cycles))
    pred_log_dQ = logk + alpha * np.log(pred_efc) + beta * np.log(dod_ref) + gamma * np.log(c_rate_ref)
    pred_dQ = np.exp(pred_log_dQ)
    pred_dQ = np.clip(pred_dQ, 1e-6, 0.4)
    pred_soh = 1 - pred_dQ
    pred_capacity = pred_soh * Q0
    pred_df = pd.DataFrame({
        "预测循环数(EFC)": pred_efc.astype(int),
        "预测SOH": pred_soh,
        "预测容量(Ah)": pred_capacity,
        "容量衰减量": pred_dQ
    })
    return pred_df

# ======================================
# ✅ 主流程函数 - 适配手动Q0+温度工况
# ======================================
def run_pipeline(csv_file,cmap: ColumnMap,cfg: FitConfig,ref_conditions: Dict[str, float], manual_Q0: Optional[float] = None):
    df = pd.read_csv(csv_file)
    df_feat, Q0, q0_source = compute_features(df, cmap, manual_Q0)
    fit = fit_life_model(df_feat, cmap, cfg)

    dod_ref = float(ref_conditions["dod"])
    c_rate_ref = float(ref_conditions["c_rate"])
    temp_c_ref = float(ref_conditions["temp_c"])
    target_soh = cfg.soh_target

    Nlife = solve_life_to_target(fit["params"], target_soh, dod_ref, c_rate_ref)
    lo, hi = bootstrap_life_ci(df_feat, cmap, cfg, dod_ref, c_rate_ref)
    pred_full_df = predict_full_life_cycles(fit, Q0, target_soh, Nlife, dod_ref, c_rate_ref)

    result = {
        "Q0_ah_est": Q0,
        "Q0_source": q0_source,
        "fit": fit,
        "ref_conditions": {"temp_c": temp_c_ref,"dod": dod_ref,"c_rate": c_rate_ref,"soh_target": target_soh},
        "life_N_point": Nlife,
        "life_N_CI95": (lo, hi),
        "feat_df": df_feat,
        "predict_full_df": pred_full_df
    }
    return result

# ======================================
# ✅ 纯净版网页界面 - 新增【手动输入Q0】核心优化 最终版
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
        st.markdown("<h4 style='color: #2980b9; border-bottom:2px solid #3498db; padding-bottom:5px'>⚙️ 核心参数配置</h4>", unsafe_allow_html=True)
        # ✅ 新增：手动输入额定容量Q0，留空则自动取最大值
        manual_Q0 = st.number_input("额定容量 Q0 (Ah)", min_value=0.1, max_value=1000.0, value=None, step=0.01, format="%.2f", help="留空则自动取实测容量最大值，已知额定值建议手动输入")
        target_soh = st.number_input("寿命终点SOH值", min_value=0.6, max_value=0.95, value=0.80, step=0.01, format="%.2f")
        temp_c_ref = st.number_input("工况温度(℃)", min_value=0.0, max_value=60.0, value=25.0, step=0.5, format="%.1f")
        dod_ref = st.number_input("放电深度(DoD)", min_value=0.0, max_value=1.0, value=1.0, step=0.01, format="%.2f")
        c_rate_ref = st.number_input("放电倍率(C-rate)", min_value=0.01, max_value=5.0, value=0.5, step=0.01, format="%.2f")

        st.markdown("<h4 style='color: #2980b9; border-bottom:2px solid #3498db; padding-bottom:5px; margin-top:20px'>📂 上传数据</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("选择CSV文件", type="csv")

        run_btn = st.button("开始预测", use_container_width=True, type="primary")

    with col2:
        if run_btn and uploaded_file is not None:
            try:
                with st.spinner("计算中..."):
                    cfg = FitConfig(soh_target=target_soh)
                    ref_conditions = {"temp_c":temp_c_ref, "dod":dod_ref, "c_rate":c_rate_ref}
                    all_result = run_pipeline(uploaded_file, cmap, cfg, ref_conditions, manual_Q0)
                    fit_params = all_result["fit"]["params"]
                    life_cycle = int(all_result["life_N_point"])
                    ci_low, ci_high = int(all_result["life_N_CI95"][0]), int(all_result["life_N_CI95"][1])
                    Q0 = all_result["Q0_ah_est"]
                    q0_source = all_result["Q0_source"]
                    pred_df = all_result["predict_full_df"]
                    feat_df = all_result["feat_df"]
                    filter_df = all_result["fit"]["filtered_df"]

                st.markdown("<h4 style='color: #2980b9; border-bottom:2px solid #3498db; padding-bottom:5px'>📊 预测结果</h4>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(f"""
                    <div style='color: #2c3e50; font-size: 14px; line-height: 1.8;'>
                    初始容量 Q0: {Q0:.3f} Ah ({q0_source}) | 有效拟合循环数: {all_result['fit']['n_used']} 个<br>
                    目标SOH: {target_soh*100:.1f}% | 工况温度: {temp_c_ref}℃ | 放电深度: {dod_ref*100:.1f}% | 放电倍率: {c_rate_ref}C<br>
                    预测总循环数: <span style='color: #e67e22; font-weight: bold; font-size:15px;'>{life_cycle}</span> 次<br>
                    95%置信区间: <span style='color: #e67e22;'>[{ci_low} ~ {ci_high}]</span> 次
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<h4 style='color: #2980b9; border-bottom:2px solid #3498db; padding-bottom:5px; margin-top:10px'>⚙️ 模型拟合参数</h4>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(f"""
                    <div style='color: #2c3e50; font-size: 13px; line-height: 1.8;'>
                    k: {fit_params['k']:.6f} | logk: {fit_params['logk']:.6f}<br>
                    α: {fit_params['alpha']:.6f} | β: {fit_params['beta_dod']:.6f} | γ: {fit_params['gamma_crate']:.6f}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<h4 style='color: #2980b9; border-bottom:2px solid #3498db; padding-bottom:5px; margin-top:10px'>📈 SOH衰减曲线</h4>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
                ax.plot(feat_df["efc"], feat_df["soh"], 'b-', linewidth=2.0, label='实测SOH', alpha=0.9)
                ax.plot(filter_df["efc"], 1-filter_df["dQ"], 'r--', linewidth=2.0, label='模型拟合SOH', alpha=0.9)
                ax.plot(pred_df["预测循环数(EFC)"], pred_df["预测SOH"], 'orange', linestyle='-.', linewidth=2.0, label='全循环预测SOH', alpha=0.9)
                ax.axhline(y=target_soh, color='#e74c3c', linestyle=':', linewidth=2, label=f'寿命终点({target_soh*100}% SOH)')
                ax.axvline(x=life_cycle, color='#f39c12', linestyle=':', linewidth=1.8, label=f'预测总寿命: {life_cycle} 循环')
                ax.set_title(f'SOH Attenuation Curve (T={temp_c_ref}℃, DoD={dod_ref}, C-rate={c_rate_ref}, Q0={Q0}Ah)', fontsize=12, fontweight='bold', color='#2c3e50')
                ax.set_xlabel("等效满充循环数 (EFC)", fontsize=11, color='#2c3e50')
                ax.set_ylabel("电芯健康状态 (SOH)", fontsize=11, color='#2c3e50')
                ax.legend(loc='upper right', framealpha=0.9, facecolor='white', edgecolor='#bdc3c7')
                ax.grid(True, alpha=0.5)
                ax.set_ylim(0.55, 1.05)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)

                st.markdown("<h4 style='color: #2980b9; border-bottom:2px solid #3498db; padding-bottom:5px; margin-top:10px'>💾 数据导出</h4>", unsafe_allow_html=True)
                real_data = feat_df[["cycle","capacity_ah","soh","dQ","c_rate","efc",cmap.temp_c,"Q0_ah"]].copy()
                real_data.rename(columns={
                    "cycle":"实测循环数","capacity_ah":"实测容量(Ah)","soh":"实测SOH","dQ":"实测衰减量",
                    "c_rate":"放电倍率","efc":"等效循环数",cmap.temp_c:"平均温度(℃)","Q0_ah":"初始容量(Ah)"
                },inplace=True)
                export_df = pd.concat([real_data, pred_df], ignore_index=True)
                csv_data = export_df.to_csv(index=False, encoding="utf-8-sig").encode('utf-8-sig')
                
                st.download_button(
                    label="下载完整预测数据",
                    data=csv_data,
                    file_name=f"电芯寿命预测结果.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )

            except:
                st.error("数据格式错误，请检查文件后重试")

        elif run_btn:
            st.warning("请先上传CSV文件")

if __name__ == "__main__":
    main()

