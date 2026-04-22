import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Coulomb Lab",
    page_icon="⚡",
    layout="wide",
)

K = 8.9875517923e9
MICRO = 1e-6
CM = 1e-2
MM = 1e-3


def coulomb_force(q1_c, q2_c, r_m):
    if r_m <= 0:
        return np.nan
    return K * abs(q1_c * q2_c) / (r_m ** 2)


def field_at_x(q1_c, x1, q2_c, x2, x):
    def contribution(q, xq, xp):
        dx = xp - xq
        if abs(dx) < 1e-12:
            return np.nan
        return K * q * dx / abs(dx) ** 3
    return contribution(q1_c, x1, x) + contribution(q2_c, x2, x)


def potential_at_x(q1_c, x1, q2_c, x2, x):
    d1 = abs(x - x1)
    d2 = abs(x - x2)
    v1 = np.nan if d1 < 1e-12 else K * q1_c / d1
    v2 = np.nan if d2 < 1e-12 else K * q2_c / d2
    return v1 + v2


def force_direction_text(q1_u, q2_u):
    return "Tolak-menolak" if q1_u * q2_u > 0 else "Tarik-menarik"


def interaction_arrow(q1_u, q2_u):
    return "←   →" if q1_u * q2_u > 0 else "→   ←"


def init_state():
    if "dataset" not in st.session_state:
        st.session_state.dataset = pd.DataFrame(columns=[
            "q1_uC", "q2_uC", "r_cm", "F_N", "1_over_r2", "arah"
        ])


init_state()

st.title("Aplikasi Pembelajaran Fisika. Hukum Coulomb")
st.caption("Level sarjana. Fokus pada hubungan gaya listrik, tanda muatan, dan jarak antar muatan.")

with st.sidebar:
    st.header("Parameter simulasi")
    q1_u = st.slider("Muatan q1 (μC)", -20.0, 20.0, 5.0, 0.5)
    q2_u = st.slider("Muatan q2 (μC)", -20.0, 20.0, -5.0, 0.5)
    r_cm = st.slider("Jarak r (cm)", 1.0, 100.0, 20.0, 1.0)
    st.divider()
    show_two_charge_field = st.checkbox("Tampilkan medan dan potensial 1D", value=True)
    x_probe = st.slider("Titik uji x (m)", -1.0, 2.0, 0.75, 0.01)
    st.markdown("Nilai muatan diubah ke Coulomb saat dihitung.")

q1_c = q1_u * MICRO
q2_c = q2_u * MICRO
r_m = r_cm * CM

F = coulomb_force(q1_c, q2_c, r_m)
arah = force_direction_text(q1_u, q2_u)
arrow = interaction_arrow(q1_u, q2_u)

col1, col2, col3, col4 = st.columns(4)
col1.metric("|F|", f"{F:.4e} N")
col2.metric("Jenis interaksi", arah)
col3.metric("r", f"{r_m:.3f} m")
col4.metric("1/r²", f"{1/(r_m**2):.3f} m⁻²")

st.subheader("Persamaan utama")
st.latex(r"F = k\frac{|q_1q_2|}{r^2}")

left, right = st.columns([1.1, 1])

with left:
    st.subheader("Representasi sistem dua muatan")
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=[0, r_m], y=[0, 0], mode="lines",
        line=dict(width=4), showlegend=False
    ))
    fig_line.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(size=26, symbol="circle"),
        text=[f"q1={q1_u:.1f} μC"], textposition="top center", name="q1"
    ))
    fig_line.add_trace(go.Scatter(
        x=[r_m], y=[0], mode="markers+text",
        marker=dict(size=26, symbol="circle"),
        text=[f"q2={q2_u:.1f} μC"], textposition="top center", name="q2"
    ))
    mid = r_m / 2
    fig_line.add_annotation(x=mid, y=0.08, text=f"r = {r_m:.3f} m", showarrow=False)
    fig_line.add_annotation(x=mid, y=-0.15, text=arrow, showarrow=False, font=dict(size=20))
    fig_line.update_layout(
        height=320,
        xaxis_title="Posisi (m)",
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_line, use_container_width=True, theme="streamlit")

with right:
    st.subheader("Interpretasi cepat")
    if q1_u == 0 or q2_u == 0:
        st.info("Salah satu muatan nol. Gaya listrik menjadi nol.")
    else:
        if q1_u * q2_u > 0:
            st.write("Kedua muatan bertanda sama. Gaya yang muncul adalah gaya tolak-menolak.")
        else:
            st.write("Kedua muatan bertanda berbeda. Gaya yang muncul adalah gaya tarik-menarik.")
        st.write("Besar gaya berbanding lurus dengan hasil kali besar kedua muatan.")
        st.write("Besar gaya berbanding terbalik dengan kuadrat jarak. Saat jarak diperbesar dua kali, gaya menjadi seperempat.")
    st.write("Gunakan slider untuk menguji pola tersebut.")

st.divider()
st.subheader("Eksperimen virtual dan pencatatan data")

exp1, exp2 = st.columns([1, 1])
with exp1:
    if st.button("Tambahkan data saat ini"):
        new_row = pd.DataFrame([{
            "q1_uC": q1_u,
            "q2_uC": q2_u,
            "r_cm": r_cm,
            "F_N": F,
            "1_over_r2": 1/(r_m**2),
            "arah": arah,
        }])
        st.session_state.dataset = pd.concat([st.session_state.dataset, new_row], ignore_index=True)
        st.success("Data ditambahkan ke tabel eksperimen.")
with exp2:
    if st.button("Hapus semua data"):
        st.session_state.dataset = st.session_state.dataset.iloc[0:0]
        st.warning("Semua data eksperimen dihapus.")

df = st.session_state.dataset.copy()
st.dataframe(df, use_container_width=True, hide_index=True)

if not df.empty:
    c1, c2 = st.columns(2)
    with c1:
        fig_f_r = px.line(df, x="r_cm", y="F_N", markers=True, title="Grafik |F| terhadap r")
        st.plotly_chart(fig_f_r, use_container_width=True, theme="streamlit")
    with c2:
        fig_f_inv = px.line(df, x="1_over_r2", y="F_N", markers=True, title="Grafik |F| terhadap 1/r²")
        st.plotly_chart(fig_f_inv, use_container_width=True, theme="streamlit")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Unduh data CSV",
        data=csv,
        file_name="data_hukum_coulomb.csv",
        mime="text/csv"
    )
else:
    st.info("Belum ada data. Tambahkan beberapa titik eksperimen untuk melihat grafik.")

if show_two_charge_field:
    st.divider()
    st.subheader("Medan listrik dan potensial listrik sepanjang sumbu x")
    st.write("Pada bagian ini, q1 ditempatkan di x = 0 m dan q2 di x = 1 m.")

    x1 = 0.0
    x2 = 1.0
    xs = np.linspace(-1.0, 2.0, 600)
    E_vals = []
    V_vals = []
    for x in xs:
        e = field_at_x(q1_c, x1, q2_c, x2, x)
        v = potential_at_x(q1_c, x1, q2_c, x2, x)
        E_vals.append(np.nan if np.isinf(e) else e)
        V_vals.append(np.nan if np.isinf(v) else v)

    df_field = pd.DataFrame({"x": xs, "E": E_vals, "V": V_vals})
    probe_E = field_at_x(q1_c, x1, q2_c, x2, x_probe)
    probe_V = potential_at_x(q1_c, x1, q2_c, x2, x_probe)

    c3, c4 = st.columns(2)
    c3.metric("E di titik uji", f"{probe_E:.4e} N/C" if np.isfinite(probe_E) else "Tak terdefinisi")
    c4.metric("V di titik uji", f"{probe_V:.4e} V" if np.isfinite(probe_V) else "Tak terdefinisi")

    fig_E = px.line(df_field, x="x", y="E", title="Medan listrik E(x)")
    fig_E.add_vline(x=0.0, line_dash="dash")
    fig_E.add_vline(x=1.0, line_dash="dash")
    fig_E.add_vline(x=x_probe, line_dash="dot")
    st.plotly_chart(fig_E, use_container_width=True, theme="streamlit")

    fig_V = px.line(df_field, x="x", y="V", title="Potensial listrik V(x)")
    fig_V.add_vline(x=0.0, line_dash="dash")
    fig_V.add_vline(x=1.0, line_dash="dash")
    fig_V.add_vline(x=x_probe, line_dash="dot")
    st.plotly_chart(fig_V, use_container_width=True, theme="streamlit")

st.divider()
st.subheader("Latihan analisis")
q_a = st.number_input("Prediksi Anda. Jika r dibuat 2 kali lebih besar, gaya menjadi berapa kali?", value=0.25)
if st.button("Cek jawaban konsep"):
    if abs(q_a - 0.25) < 1e-9:
        st.success("Benar. Karena F berbanding terbalik dengan r kuadrat, maka F baru = 1/4 F awal.")
    else:
        st.error("Belum tepat. Gunakan relasi F ∝ 1/r².")

st.subheader("Catatan untuk dosen")
st.write(
    "Aplikasi ini cocok untuk demonstrasi kelas, tugas eksplorasi mandiri, atau praktikum komputasi sederhana. "
    "Mahasiswa dapat diminta menguji linearitas grafik |F| terhadap 1/r², membandingkan interaksi tarik dan tolak, "
    "serta menghubungkan tanda gradien medan dengan tanda muatan."
)
