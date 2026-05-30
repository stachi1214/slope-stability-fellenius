import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from geometry import (
    create_slope_geometry,
    create_circle_geometry,
    find_ground_circle_intersections,
    create_slice_boundaries,
    create_slices
)

from fellenius import calculate_fellenius_factor_of_safety


st.set_page_config(
    page_title="円弧すべり安全率計算",
    layout="wide"
)

st.title("円弧すべり安全率計算アプリ")
st.caption("フェルニウス法による斜面安定解析")


# -------------------------
# 入力欄
# -------------------------
st.sidebar.header("入力条件")

H = st.sidebar.number_input(
    "斜面高さ H (m)",
    min_value=0.1,
    value=10.0,
    step=0.5
)

m = st.sidebar.number_input(
    "斜面勾配 1:m",
    min_value=0.1,
    value=1.5,
    step=0.1
)

gamma = st.sidebar.number_input(
    "単位体積重量 γ (kN/m³)",
    min_value=0.1,
    value=18.0,
    step=0.5
)

c = st.sidebar.number_input(
    "有効粘着力 c' (kPa)",
    min_value=0.0,
    value=10.0,
    step=1.0
)

phi_deg = st.sidebar.number_input(
    "有効摩擦角 φ' (degree)",
    min_value=0.0,
    max_value=60.0,
    value=30.0,
    step=1.0
)

xc = st.sidebar.number_input(
    "円中心 x座標 xc (m)",
    value=5.0,
    step=0.5
)

yc = st.sidebar.number_input(
    "円中心 y座標 yc (m)",
    value=15.0,
    step=0.5
)

R = st.sidebar.number_input(
    "円の半径 R (m)",
    min_value=0.1,
    value=15.0,
    step=0.5
)

n_slices = st.sidebar.number_input(
    "スライス数",
    min_value=2,
    max_value=200,
    value=20,
    step=1
)


# -------------------------
# 幾何データ作成
# -------------------------
x_ground, y_ground = create_slope_geometry(H, m)
x_circle, y_circle = create_circle_geometry(xc, yc, R)
intersections = find_ground_circle_intersections(H, m, xc, yc, R)

slice_boundaries = create_slice_boundaries(
    H=H,
    m=m,
    xc=xc,
    yc=yc,
    R=R,
    intersections=intersections,
    n_slices=int(n_slices)
)

slices = create_slices(
    H=H,
    m=m,
    xc=xc,
    yc=yc,
    R=R,
    intersections=intersections,
    n_slices=int(n_slices),
    gamma=gamma
)

# -------------------------
# 図の作成
# -------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x_ground,
    y=y_ground,
    mode="lines",
    name="地表面",
    line=dict(width=3)
))

fig.add_trace(go.Scatter(
    x=x_circle,
    y=y_circle,
    mode="lines",
    name="すべり円",
    line=dict(width=2)
))

if len(intersections) > 0:
    fig.add_trace(go.Scatter(
        x=[p["x"] for p in intersections],
        y=[p["y"] for p in intersections],
        mode="markers",
        name="交点",
        marker=dict(size=10)
    ))

if len(slice_boundaries) > 0:
    x_lines = []
    y_lines = []

    for b in slice_boundaries:
        x_lines.extend([b["x"], b["x"], None])
        y_lines.extend([b["y_circle"], b["y_ground"], None])

    fig.add_trace(go.Scatter(
        x=x_lines,
        y=y_lines,
        mode="lines",
        name="スライス境界",
        line=dict(width=1, dash="dot")
    ))

fig.update_layout(
    width=800,
    height=600,
    xaxis=dict(
        title="x (m)",
        range=[-1.2 * H, m * H + 1.2 * H]
    ),
    yaxis=dict(
        title="y (m)",
        scaleanchor="x",
        scaleratio=1
    ),
    showlegend=True
)


# -------------------------
# 安全率計算
# -------------------------
fellenius_result = calculate_fellenius_factor_of_safety(
    slices=slices,
    c=c,
    phi_deg=phi_deg
)

Fs = fellenius_result["Fs"]
calculated_slices = fellenius_result["slices"]


# -------------------------
# 画面表示
# -------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("計算結果")

    st.subheader("幾何条件チェック")

    if len(intersections) == 0:
        st.error("すべり円が地表面と交わっていません。円中心または半径を見直してください。")

    elif len(intersections) == 1:
        st.warning("すべり円が地表面に1点で接しています。円弧すべり面としては不十分です。")

    elif len(intersections) == 2:
        st.success("すべり円が地表面と2点で交わっています。")

    else:
        st.warning(
            f"すべり円が地表面と {len(intersections)} 点で交わっています。"
            " 解析対象とする円弧の選択ルールを追加する必要があります。"
        )

    for i, p in enumerate(intersections, start=1):
        st.write(
            f"交点 {i}: x = {p['x']:.3f} m, y = {p['y']:.3f} m, "
            f"位置 = {p['segment']}"
        )

    if fellenius_result["sum_driving"] <= 0:
        st.error(
            "滑動項の合計が0以下です。"
            " すべり方向または円弧形状が不自然な可能性があります。"
        )

    if Fs is None:
        st.warning("安全率を計算できません。滑動項がゼロに近い可能性があります。")
    else:
        st.metric("安全率 Fs", f"{Fs:.3f}")

    st.write(f"抵抗項合計 = {fellenius_result['sum_resisting']:.3f} kN/m")
    st.write(f"滑動項合計 = {fellenius_result['sum_driving']:.3f} kN/m")


    st.subheader("入力条件の確認")
    st.write(f"斜面高さ H = {H} m")
    st.write(f"斜面勾配 = 1:{m}")
    st.write(f"単位体積重量 γ = {gamma} kN/m³")
    st.write(f"有効粘着力 c' = {c} kPa")
    st.write(f"有効摩擦角 φ' = {phi_deg}°")
    st.write(f"円中心 = ({xc}, {yc}) m")
    st.write(f"円半径 R = {R} m")
    st.write(f"スライス数 = {n_slices}")

    st.subheader("スライス分割")

    if len(slice_boundaries) == 0:
        st.warning("スライス分割を作成できません。交点が2点であるか確認してください。")
    else:
        st.success(f"{int(n_slices)} 個のスライスに分割しました。")
        st.write(f"スライス境界線数：{len(slice_boundaries)} 本")

    st.subheader("スライス重量・フェルニウス法計算表")

    if len(calculated_slices) == 0:
        st.warning("スライス計算表を作成できません。")
    else:
        total_area = sum(s["area"] for s in calculated_slices)
        total_weight = sum(s["weight"] for s in calculated_slices)

        st.success("スライス重量とフェルニウス法の計算項目を計算しました。")
        st.write(f"総断面積 A = {total_area:.3f} m²")
        st.write(f"総重量 W = {total_weight:.3f} kN/m")

        df_slices = pd.DataFrame(calculated_slices)

        st.dataframe(
            df_slices[
                [
                    "i",
                    "x_left",
                    "x_right",
                    "width",
                    "area",
                    "weight",
                    "base_length",
                    "alpha_deg",
                    "cohesion_resistance",
                    "friction_resistance",
                    "resisting",
                    "driving"
                ]
            ],
            use_container_width=True
        )