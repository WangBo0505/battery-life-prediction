# ===================== 全局配置 - 网络图片LOGO版（零报错） =====================
st.set_page_config(
    page_title="储能电池全生命周期预测系统",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅✅✅ 替换这里：把你的LOGO在线链接粘贴到下面的引号里
LOGO_URL = "https://www.ptl-global.com/cn/img/logo.png"

st.markdown(f"""
    <style>
        .fixed-logo {{
            position: fixed;
            top: 30px;
            right: 30px;
            width: 120px;
            z-index: 9999;
        }}
    </style>
    <img src="{LOGO_URL}" class="fixed-logo" alt="logo">
""", unsafe_allow_html=True)
