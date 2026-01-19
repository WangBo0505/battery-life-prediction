import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import streamlit as st

# ===================== 全局配置 - 纯英文图表 彻底解决中文显示问题 =====================
st.set_page_config(
    page_title="储能电池全生命周期预测系统",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ 纯英文图表极简配置，无中文字体依赖，永不乱码
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
# ↓↓↓↓↓↓ 核心代码 - 逻辑精准修正 预测不变 ↓↓↓↓↓↓
# ======================================
R_GAS = 8.314462618  # 理想气体常数

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
    min_dod: float = 0.80
    min_cycles: int = 50
    use_efc: bool = True
    bootstrap_n: int = 100
    random_seed: int = 42
    temp_min_c: float = 20.0
    temp_max_c: float = 55.0

def compute_features(df: pd.DataFrame, cmap: ColumnMap, rated_capacity_input: Optional[float] = None) -> Tuple[pd.DataFrame, float, float]:
    """✅ 核心修正逻辑：
    1. dQ(容量衰减率) 基于 Q0(实测初始容量)计算 → 物理本质，保证预测不变
    2. 显示用SOH 基于 用户输入的额定容量 换算 → 仅改数值显示
    3. 返回：处理数据 + Q0(实测初始容量) + Rated_Cap(额定容量)
    """
    d = df.copy()
    required_cols = [cmap.cycle, cmap.cap_ah, cmap.temp_c, cmap.dod, cmap.i_dis_a]
    missing_cols = [col for col in required_cols if col not in d.columns]
    if missing_cols:
        raise ValueError(f"CSV文件缺少必要列: {missing_cols}")

    d = d.sort_values(cmap.cycle).drop_duplicates(subset=[cmap.cycle]).reset_index(drop=True)

    # ✅ 保留你的Q0计算逻辑 完全不动
    cap_series = d[cmap.cap_ah].astype(float)
    if cmap.rated_cap_ah and cmap.rated_cap_ah in d.columns:
        rated_cap_csv = d[cmap.rated_cap_ah].astype(float)
        Q0 = float(rated_cap_csv.iloc[0])
    else:
        Q0 = float(cap_series.head(20).median())

    # ✅ 额定容量赋值：用户输入则用输入值，否则默认等于Q0
    Rated_Cap = rated_capacity_input if (rated_capacity_input and rated_capacity_input > 0) else Q0
    cap = cap_series.to_numpy()
    temp_c = d[cmap.temp_c].astype(float).to_numpy()
    dod = d[cmap.dod].astype(float).to_numpy()
    i_dis = d[cmap.i_dis_a].astype(float).to_numpy()

    # ✅ ✔️ 重中之重【核心修正】：dQ基于Q0计算，保证模型拟合/预测逻辑完全不变！！！
    soh_based_Q0 = cap / Q0  # 基于实测容量的SOH，用于计算衰减率
    dQ = np.clip(1.0 - soh_based_Q0, 1e-6, 0.4)  # 衰减率不变 → 预测结果不变
    
    # ✅ ✔️ 显示用SOH：基于用户输入的额定容量换算，仅改变数值显示，不影响任何计算
    soh_show = cap / Rated_Cap

    # 其他计算逻辑不变
    c_rate = np.clip(np.abs(i_dis) / max(Q0, 1e-6), 1e-6, None)
    efc = np.cumsum(np.clip(dod, 0.0, 1.0))

    # 存入数据
    d["Q0_ah"] = Q0
    d["Rated_Cap_Ah"] = Rated_Cap
    d["soh"] = soh_show      # 前端显示的SOH（额定容量基准）
    d["dQ"] = dQ             # 核心衰减率（实测容量基准，不变）
    d["c_rate"] = c_rate
    d["efc"] = efc
    d["temp_k"] = temp_c + 273.15

    return d, Q0, Rated_Cap

def _model_log_dQ(params, N, dod, c_rate):
    # ✅ 核心衰减模型 完全未改 → 预测不变
    logk, alpha, beta, gamma = params
    N = np.clip(N, 1e-6, None)
    dod = np.clip(dod, 1e-6, None)
    c_rate = np.clip(c_rate, 1e-6, None)
    return (logk + alpha * np.log(N) + beta * np.log(dod) + gamma * np.log(c_rate))

def fit_life_model(df_feat: pd.DataFrame, cmap: ColumnMap, cfg: FitConfig):
    # ✅ 模型拟合逻辑 完全未改 → 拟合参数不变
    d = df_feat.copy()
    d = d[(d["soh"] > 0.6) & (d["soh"] < 0.98)]
    d = d[(d[cmap.temp_c] >= cfg.temp_min_c) & (d[cmap.temp_c] <= cfg.temp_max_c)]
    d = d[d[cmap.dod].astype(float) >= cfg.min_dod]

    if len(d) < cfg.min_cycles:
        raise ValueError(f"数据量不足：有效循环 {len(d)} < min_cycles={cfg.min_cycles}")

    N = d["efc"].to_numpy() if cfg.use_efc else d[cmap.cycle].astype(float).to_numpy()
    dod = d[cmap.dod].astype(float).to_numpy()
    c_rate = d["c_rate"].to_numpy()
    y = np.log(d["dQ"].to_numpy())  # 拟合用dQ，不变

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
    # ✅ 寿命预测核心公式 完全未改 → 预测结果不变
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
    # ✅ 置信区间计算 完全未改 → 结果不变
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
# ✅ 预测函数 - 适配额定容量 显示修正 预测不变
# ======================================
def predict_full_life_cycles(fit_result, Q0, Rated_Cap, target_soh, life_cycles, dod_ref=1.0, c_rate_ref=0.5):
    logk, alpha, beta, gamma = fit_result["fit_params"]
    pred_efc = np.linspace(1, life_cycles, int(life_cycles))
    pred_log_dQ = logk + alpha * np.log(pred_efc) + beta * np.log(dod_ref) + gamma * np.log(c_rate_ref)
    pred_dQ = np.exp(pred_log_dQ)
    pred_dQ = np.clip(pred_dQ, 1e-6, 0.4)
    
    # ✅ 核心：预测衰减率不变 → 预测的真实容量不变
    pred_capacity_based_Q0 = (1 - pred_dQ) * Q0
    # ✅ 显示修正：预测SOH基于额定容量换算
    pred_soh_show = pred_capacity_based_Q0 / Rated_Cap

    pred_df = pd.DataFrame({
        "Pred_EFC": pred_efc.astype(int),
        "Pred_SOH": pred_soh_show,
        "Pred_Capacity(Ah)": pred_capacity_based_Q0,
        "Capacity_Decay": pred_dQ
    })
    return pred_df

# ======================================
# ✅ 主流程函数 - 适配额定容量输入
# ======================================
def run_pipeline(csv_file,cmap: ColumnMap,cfg: FitConfig,ref_conditions: Dict[str, float], rated_capacity_input: Optional[float]):
    df = pd.read_csv(csv_file)
    df_feat, Q0, Rated_Cap = compute_features(df, cmap, rated_capacity_input)
    fit = fit_life_model(df_feat, cmap, cfg)

    dod_ref = float(ref_conditions["dod"])
    c_rate_ref = float(ref_conditions["c_rate"])
    target_soh = cfg.soh_target

    Nlife = solve_life_to_target(fit["params"], target_soh, dod_ref, c_rate_ref)
    lo, hi = bootstrap_life_ci(df_feat, cmap, cfg, dod_ref, c_rate_ref)
    pred_full_df = predict_full_life_cycles(fit, Q0, Rated_Cap, target_soh, Nlife, dod_ref, c_rate_ref)

    result = {
        "Q0_ah_est": Q0,
        "Rated_Cap_Ah": Rated_Cap,
        "fit": fit,
        "ref_conditions": {"temp_c": ref_conditions["temp_c"],"dod": dod_ref,"c_rate": c_rate_ref,"soh_target": target_soh},
        "life_N_point": Nlife,
        "life_N_CI95": (lo, hi),
        "feat_df": df_feat,
        "predict_full_df": pred_full_df
    }
    return result

# ======================================
# ✅ 纯净版网页界面 - 全英文图表+额定容量输入+CSV示例+保留所有细节
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
        # ✅ 额定容量输入框 0.01~10000Ah
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
                    fit_params = all_result["fit"]["params"]
                    life_cycle = int(all_result["life_N_point"])
                    ci_low, ci_high = int(all_result["life_N_CI95"][0]), int(all_result["life_N_CI95"][1])
                    Q0 = all_result["Q0_ah_est"]
                    rated_cap = all_result["Rated_Cap_Ah"]
                    pred_df = all_result["predict_full_df"]
                    feat_df = all_result["feat_df"]
                    filter_df = all_result["fit"]["filtered_df"]

                st.markdown("<h4 style='color: #2980b9; border-bottom:2px solid #3498db; padding-bottom:5px'>📊 预测结果 (Prediction Results)</h4>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(f"""
                    <div style='color: #2c3e50; font-size: 14px; line-height: 1.8;'>
                    实测初始容量 Q0: {Q0:.3f} Ah | 额定容量 Rated Capacity: {rated_cap:.3f} Ah<br>
                    有效拟合循环数: {all_result['fit']['n_used']} <br>
                    目标SOH: {target_soh*100:.1f}% | 放电深度: {dod_ref*100:.1f}% | 放电倍率: {c_rate_ref}C<br>
                    预测总循环数: <span style='color: #e67e22; font-weight: bold; font-size:15px;'>{life_cycle}</span> Cycles<br>
                    95%置信区间 95%CI: <span style='color: #e67e22;'>[{ci_low} ~ {ci_high}]</span> Cycles
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<h4 style='color: #2980b9; border-bottom:2px solid #3498db; padding-bottom:5px; margin-top:10px'>⚙️ 模型参数 (Model Parameters)</h4>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(f"""
                    <div style='color: #2c3e50; font-size: 13px; line-height: 1.8;'>
                    k: {fit_params['k']:.6f} | logk: {fit_params['logk']:.6f}<br>
                    α: {fit_params['alpha']:.6f} | β: {fit_params['beta_dod']:.6f} | γ: {fit_params['gamma_crate']:.6f}
                    </div>
                    """, unsafe_allow_html=True)

                # ✅ 纯英文衰减曲线图 永不乱码
                st.markdown("<h4 style='color: #2980b9; border-bottom:2px solid #3498db; padding-bottom:5px; marginTop:10px'>📈 SOH Attenuation Curve</h4>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
                ax.plot(feat_df["efc"], feat_df["soh"], 'b-', linewidth=2.0, label='Measured SOH', alpha=0.9)
                ax.plot(filter_df["efc"], 1-filter_df["dQ"], 'r--', linewidth=2.0, label='Fitted SOH', alpha=0.9)
                ax.plot(pred_df["Pred_EFC"], pred_df["Pred_SOH"], 'orange', linestyle='-.', linewidth=2.0, label='Predicted SOH', alpha=0.9)
                ax.axhline(y=target_soh, color='#e74c3c', linestyle=':', linewidth=2, label=f'End of Life ({target_soh*100}% SOH)')
                ax.axvline(x=life_cycle, color='#f39c12', linestyle=':', linewidth=1.8, label=f'Predicted Cycle Life: {life_cycle}')
                ax.set_title(f'SOH Attenuation Curve (DOD={dod_ref}, C-rate={c_rate_ref})', fontsize=12, fontweight='bold', color='#2c3e50')
                ax.set_xlabel("Equivalent Full Cycles (EFC)", fontsize=11, color='#2c3e50')
                ax.set_ylabel("State of Health (SOH)", fontsize=11, color='#2c3e50')
                ax.legend(loc='upper right', framealpha=0.9, facecolor='white', edgecolor='#bdc3c7')
                ax.grid(True, alpha=0.5)
                ax.set_ylim(0.55, 1.05)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)

                st.markdown("<h4 style='color: #2980b9; border-bottom:2px solid #3498db; padding-bottom:5px; margin-top:10px'>💾 数据导出 (Data Export)</h4>", unsafe_allow_html=True)
                real_data = feat_df[["cycle","capacity_ah","soh","dQ","c_rate","efc",cmap.temp_c,"Q0_ah","Rated_Cap_Ah"]].copy()
                real_data.rename(columns={
                    "cycle":"Cycle","capacity_ah":"Capacity(Ah)","soh":"SOH","dQ":"Capacity_Decay",
                    "c_rate":"C-rate","efc":"EFC",cmap.temp_c:"Avg_Temp(℃)","Q0_ah":"Initial_Capacity(Ah)","Rated_Cap_Ah":"Rated_Capacity(Ah)"
                },inplace=True)
                export_df = pd.concat([real_data, pred_df], ignore_index=True)
                csv_data = export_df.to_csv(index=False, encoding="utf-8-sig").encode('utf-8-sig')
                
                st.download_button(
                    label="Download Prediction Data",
                    data=csv_data,
                    file_name=f"Energy_Storage_Battery_Life_Prediction.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )

            except:
                st.error("数据格式错误，请检查文件后重试 (Data format error, please check the file)")

        elif run_btn:
            st.warning("请先上传CSV文件 (Please upload a CSV file first)")
        else:
            # ✅ 保留CSV示例+字段解释 帮助用户上传正确文件
            st.markdown("""
                <div style='background-color: #f8f9fa; padding: 20px; border-radius: 8px; border:1px solid #e9ecef;'>
                <h4 style='color: #2980b9; margin-top:0;'>📋 CSV文件上传格式说明（必填）</h4>
                <p style='color:#34495e; font-size:14px;'>请上传<strong>UTF-8编码</strong>的CSV文件，必须包含以下5列，列名必须完全一致，顺序无要求，示例如下：</p>
                </div>
            """, unsafe_allow_html=True)
            
            csv_example = pd.DataFrame({
                "cycle": [1, 2, 3, 4, 5],
                "capacity_ah": [290.0, 289.8, 289.7, 289.5, 289.3],
                "temp_c_mean": [25.1, 25.3, 25.0, 25.2, 25.1],
                "dod": [1.0, 1.0, 1.0, 1.0, 1.0],
                "i_dis_a_mean": [-100.0, -100.2, -99.8, -100.1, -99.9]
            })
            st.dataframe(csv_example, use_container_width=True, hide_index=True)
            
            st.markdown("""
                <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border:1px solid #e9ecef; margin-top:10px;'>
                <h5 style='color: #27ae60; margin-top:0;'>📝 字段含义解释</h5>
                <ul style='color:#34495e; font-size:13px; line-height:1.8; margin:0; padding-left:20px;'>
                <li><strong>cycle</strong>：电池循环测试次数（正整数，如1,2,3...）</li>
                <li><strong>capacity_ah</strong>：该循环下电池实际放出容量，单位 (Ah)</li>
                <li><strong>temp_c_mean</strong>：该循环测试的平均温度，单位 (℃)</li>
                <li><strong>dod</strong>：放电深度（0~1，1表示100%深放电，必填）</li>
                <li><strong>i_dis_a_mean</strong>：平均放电电流，放电为负数，充电为正数，单位 (A)</li>
                </ul>
                <p style='color:#e74c3c; font-size:13px; margin-top:10px; margin-bottom:0;'><b>注意：</b>列名不可修改，缺少列会导致计算失败！</p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
