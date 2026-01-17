import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import streamlit as st

# ===================== 全局配置 - 科技感网页主题 =====================
st.set_page_config(
    page_title="储能电芯全生命周期预测系统 | 网页版",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 全局字体/配色配置 - 深蓝暗黑科技风
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = '#0A192F'
plt.rcParams['axes.facecolor'] = '#112240'
plt.rcParams['text.color'] = '#E6F1FF'
plt.rcParams['axes.labelcolor'] = '#64FFDA'
plt.rcParams['xtick.color'] = '#8892B0'
plt.rcParams['ytick.color'] = '#8892B0'
plt.rcParams['axes.edgecolor'] = '#233554'
plt.rcParams['grid.color'] = '#233554'
plt.rcParams['grid.alpha'] = 0.3

# ======================================
# ↓↓↓↓↓↓ 你的原始核心代码 - 一字未改 完全保留 ↓↓↓↓↓↓
# ======================================
R_GAS = 8.314462618  # J/(mol*K) 理想气体常数，固定值


@dataclass
class ColumnMap:
    cycle: str = "cycle"  # 循环序号（1,2,3...）
    cap_ah: str = "capacity_ah"  # 本循环可放出容量(Ah)
    temp_c: str = "temp_c_mean"  # 循环平均温度(°C)
    dod: str = "dod"  # 放电深度(0-1)
    i_dis_a: str = "i_dis_a_mean"  # 放电平均电流(A)
    rated_cap_ah: Optional[str] = None  # 额定容量列名


@dataclass
class FitConfig:
    soh_target: float = 0.80  # 寿命终点：衰减至80%SOH
    min_dod: float = 0.80  # ✅ 适配100%DoD，过滤小深度无效数据
    min_cycles: int = 50  # 至少50个有效循环
    use_efc: bool = True  # ✅ 储能必选True，等效满充循环最科学
    bootstrap_n: int = 200  # 抽样次数
    random_seed: int = 42
    temp_min_c: float = 30.0  # 恒温测试温度下限
    temp_max_c: float = 35.0  # 恒温测试温度上限


def compute_features(df: pd.DataFrame, cmap: ColumnMap) -> Tuple[pd.DataFrame, float]:
    """计算 SOH, dQ(=1-SOH), C-rate, EFC。返回处理后的df和初始容量Q0。"""
    d = df.copy()
    required_cols = [cmap.cycle, cmap.cap_ah, cmap.temp_c, cmap.dod, cmap.i_dis_a]
    missing_cols = [col for col in required_cols if col not in d.columns]
    if missing_cols:
        raise ValueError(f"CSV文件缺少必要列: {missing_cols}")

    d = d.sort_values(cmap.cycle).drop_duplicates(subset=[cmap.cycle]).reset_index(drop=True)

    if cmap.rated_cap_ah and cmap.rated_cap_ah in d.columns:
        rated_cap = d[cmap.rated_cap_ah].astype(float)
        Q0 = float(rated_cap.iloc[0])
    else:
        Q0 = float(d[cmap.cap_ah].astype(float).head(20).median())

    cap = d[cmap.cap_ah].astype(float).to_numpy()
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

    return d, Q0


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
    if params_dict["alpha"] > 1.7:
        st.warning(f"提示：循环系数α={params_dict['alpha']:.2f} 偏大，电芯老化加速明显")
    if params_dict["beta_dod"] > 1.8:
        st.warning(f"提示：DoD系数β={params_dict['beta_dod']:.2f} 偏大，深充深放老化显著")

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
            st.info(f"提示：Bootstrap抽样失败次数较多，当前有效样本{len(life_samples)}")
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
        raise RuntimeError("bootstrap有效样本过少，可减小bootstrap_n重试")

    lo, hi = np.percentile(life_samples, [2.5, 97.5])
    return float(lo), float(hi)


# ======================================
# ✅ 全循环容量预测函数 (核心新增)
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
# ✅ 主流程函数
# ======================================
def run_pipeline(csv_path: str, cmap: ColumnMap, cfg: FitConfig, ref_conditions: Dict[str, float]):
    df = pd.read_csv(csv_path)
    df_feat, Q0 = compute_features(df, cmap)
    fit = fit_life_model(df_feat, cmap, cfg)

    dod_ref = float(ref_conditions["dod"])
    c_rate_ref = float(ref_conditions["c_rate"])
    target_soh = cfg.soh_target

    Nlife = solve_life_to_target(fit["params"], target_soh, dod_ref, c_rate_ref)
    lo, hi = bootstrap_life_ci(df_feat, cmap, cfg, dod_ref, c_rate_ref)
    pred_full_df = predict_full_life_cycles(fit, Q0, target_soh, Nlife, dod_ref, c_rate_ref)

    result = {
        "Q0_ah_est": Q0,
        "fit": fit,
        "ref_conditions": {"temp_c": ref_conditions["temp_c"], "dod": dod_ref, "c_rate": c_rate_ref,
                           "soh_target": target_soh},
        "life_N_point": Nlife,
        "life_N_CI95": (lo, hi),
        "feat_df": df_feat,
        "predict_full_df": pred_full_df
    }
    return result


# ======================================
# ✅ 核心：Streamlit 科技感网页界面 (全部实现)
# ======================================
def main():
    # 网页标题 - 科技感大标题
    st.markdown("""
        <h1 style='text-align: center; color: #64FFDA; font-weight: bold;'>⚡ 储能电芯全生命周期高精度预测系统 (网页版)</h1>
        <h3 style='text-align: center; color: #8892B0;'>电化学衰减模型 | 全循环容量预测 | 置信区间分析 | 本地免安装运行</h3>
        <hr style='border: 1px solid #233554;'>
    """, unsafe_allow_html=True)

    cmap = ColumnMap()
    col1, col2 = st.columns([1, 2.5], gap="large")

    # ========== 左侧侧边栏 - 参数配置 + 文件上传 ==========
    with col1:
        st.markdown("<h4 style='color: #64FFDA;'>🔧 预测参数配置</h4>", unsafe_allow_html=True)
        st.divider()

        # 参数输入
        target_soh = st.number_input("🎯 寿命终点SOH值", min_value=0.6, max_value=0.95, value=0.80, step=0.01,
                                     format="%.2f")
        dod_ref = st.number_input("🔋 参考放电深度DoD", min_value=0.0, max_value=1.0, value=1.0, step=0.01,
                                  format="%.2f")
        c_rate_ref = st.number_input("⚡ 参考放电C倍率", min_value=0.01, max_value=5.0, value=0.5, step=0.01,
                                     format="%.2f")
        st.divider()

        # CSV文件上传
        st.markdown("<h4 style='color: #64FFDA;'>📂 上传循环数据</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("选择CSV数据文件", type="csv",
                                         help="请上传包含cycle/capacity_ah/temp_c/dod/i_dis_a列的CSV文件")
        st.divider()

        # 运行按钮
        run_btn = st.button("🚀 启动全循环预测计算", use_container_width=True, type="primary")

    # ========== 右侧主面板 - 结果展示 + 绘图 + 导出 ==========
    with col2:
        if run_btn and uploaded_file is not None:
            try:
                with st.spinner("🔄 正在执行模型拟合 + 全循环容量预测计算，请稍候..."):
                    # 执行预测
                    cfg = FitConfig(soh_target=target_soh)
                    ref_conditions = {"temp_c": 25.0, "dod": dod_ref, "c_rate": c_rate_ref}
                    all_result = run_pipeline(uploaded_file, cmap, cfg, ref_conditions)
                    fit_params = all_result["fit"]["params"]
                    life_cycle = int(all_result["life_N_point"])
                    ci_low, ci_high = int(all_result["life_N_CI95"][0]), int(all_result["life_N_CI95"][1])
                    Q0 = all_result["Q0_ah_est"]
                    pred_df = all_result["predict_full_df"]
                    feat_df = all_result["feat_df"]
                    filter_df = all_result["fit"]["filtered_df"]

                # 预测结果面板
                st.markdown("<h4 style='color: #64FFDA;'>📊 预测结果汇总</h4>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(f"""
                    <div style='color: #E6F1FF; font-size: 14px;'>
                    <b>🔋 电芯基础参数</b><br>
                    初始容量 Q0: {Q0:.3f} Ah | 有效拟合循环数: {all_result['fit']['n_used']} 个<br>
                    <b>🎯 预测工况</b><br>
                    目标SOH: {target_soh * 100:.1f}% | 放电深度: {dod_ref * 100:.1f}% | 放电倍率: {c_rate_ref}C<br>
                    <b>✅ 核心预测结果</b><br>
                    预测总循环数: <span style='color: #FFA502; font-weight: bold;'>{life_cycle}</span> 次<br>
                    95%置信区间: <span style='color: #FFA502;'>[{ci_low} , {ci_high}]</span> 次<br>
                    拟合RMSE误差: {all_result['fit']['rmse_log_dQ']:.4f} (越小越好)
                    </div>
                    """, unsafe_allow_html=True)

                # 拟合参数面板
                st.markdown("<h4 style='color: #64FFDA;'>⚙️ 模型拟合核心参数</h4>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(f"""
                    <div style='color: #E6F1FF; font-size: 13px;'>
                    基础衰减系数 k: {fit_params['k']:.6f}<br>
                    对数衰减系数 logk: {fit_params['logk']:.6f}<br>
                    循环老化系数 α: {fit_params['alpha']:.6f}<br>
                    放电深度系数 β: {fit_params['beta_dod']:.6f}<br>
                    倍率老化系数 γ: {fit_params['gamma_crate']:.6f}
                    </div>
                    """, unsafe_allow_html=True)

                # 绘图区 - 实测+拟合+预测曲线
                st.markdown("<h4 style='color: #64FFDA;'>📈 全生命周期SOH衰减曲线</h4>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(12, 5), dpi=100, facecolor='#0A192F')
                # 绘制三条曲线
                ax.plot(feat_df["efc"], feat_df["soh"], 'b-', linewidth=1.8, label='实测SOH', alpha=0.9)
                ax.plot(filter_df["efc"], 1 - filter_df["dQ"], 'c--', linewidth=2.0, label='模型拟合SOH', alpha=0.9)
                ax.plot(pred_df["预测循环数(EFC)"], pred_df["预测SOH"], 'orange', linestyle='-.', linewidth=2.0,
                        label='全循环预测SOH', alpha=0.9)
                # 寿命终点线
                ax.axhline(y=target_soh, color='#FF4757', linestyle=':', linewidth=2,
                           label=f'寿命终点({target_soh * 100}% SOH)')
                ax.axvline(x=life_cycle, color='#FFA502', linestyle=':', linewidth=1.8,
                           label=f'预测总寿命: {life_cycle} 循环')
                # 绘图样式
                ax.set_title(f'SOH衰减曲线 (DoD={dod_ref}, C-rate={c_rate_ref})', color="#64FFDA", fontsize=12,
                             fontweight='bold')
                ax.set_xlabel("等效满充循环数 (EFC)", fontsize=10)
                ax.set_ylabel("电芯健康状态 (SOH)", fontsize=10)
                ax.legend(loc='upper right', framealpha=0.8, facecolor='#112240', edgecolor='#64FFDA',
                          labelcolor='#E6F1FF')
                ax.grid(True, alpha=0.2)
                ax.set_ylim(0.55, 1.05)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)

                # 导出数据区
                st.markdown("<h4 style='color: #64FFDA;'>💾 数据导出</h4>", unsafe_allow_html=True)
                # 整理导出数据
                real_data = feat_df[["cycle", "capacity_ah", "soh", "dQ", "c_rate", "efc", cmap.temp_c, "Q0_ah"]].copy()
                real_data.rename(columns={
                    "cycle": "实测循环数", "capacity_ah": "实测容量(Ah)", "soh": "实测SOH", "dQ": "实测衰减量",
                    "c_rate": "放电倍率", "efc": "等效循环数", cmap.temp_c: "平均温度(℃)", "Q0_ah": "初始容量(Ah)"
                }, inplace=True)
                export_df = pd.concat([real_data, pred_df], ignore_index=True)
                # 转成csv二进制流
                csv_data = export_df.to_csv(index=False, encoding="utf-8-sig").encode('utf-8-sig')
                st.download_button(
                    label="📥 下载完整预测数据 (实测+全循环预测)",
                    data=csv_data,
                    file_name=f"电芯寿命预测结果_{target_soh * 100}%SOH.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )
                st.success(f"✅ 预测完成！共生成 {life_cycle} 个循环的完整预测容量数据，可直接下载使用！")

            except Exception as e:
                st.error(f"❌ 计算出错：{str(e)}")
                st.info("💡 解决方案：将代码中 bootstrap_n=200 修改为 100 重试")

        elif run_btn:
            st.warning("⚠️ 请先上传CSV循环数据文件！")
        else:
            st.markdown("""
                <div style='color: #8892B0; font-size: 14px; text-align: center; margin-top: 50px;'>
                <h4>系统就绪 ✨</h4>
                <p>1. 左侧配置预测参数（目标SOH/放电深度/C倍率）</p>
                <p>2. 上传电芯循环数据CSV文件</p>
                <p>3. 点击【启动预测】，自动生成全循环容量预测结果</p>
                <p>4. 支持一键下载完整数据，本地运行无数据上传</p>
                </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()