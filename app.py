import io
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Coulomb Studio 2D", page_icon="⚡", layout="wide")

K = 8.9875517923e9
MICRO = 1e-6


def init_state():
    if "exp_data" not in st.session_state:
        st.session_state.exp_data = pd.DataFrame(columns=[
            "label", "q_uC", "x_m", "y_m", "Fx_N", "Fy_N", "F_resultan_N"
        ])
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False


def safe_norm(vec):
    n = np.linalg.norm(vec)
    return n if n > 1e-12 else np.nan


def net_force_on_charge(charges_c, positions, index):
    ri = positions[index]
    qi = charges_c[index]
    total = np.zeros(2)
    for j, (qj, rj) in enumerate(zip(charges_c, positions)):
        if j == index:
            continue
        diff = ri - rj
        dist = np.linalg.norm(diff)
        if dist < 1e-12:
            continue
        total += K * qi * qj * diff / (dist ** 3)
    return total


def electric_field_at_point(point, charges_c, positions):
    total = np.zeros(2)
    for q, r in zip(charges_c, positions):
        diff = point - r
        dist = np.linalg.norm(diff)
        if dist < 1e-12:
            continue
        total += K * q * diff / (dist ** 3)
    return total


def electric_potential_at_point(point, charges_c, positions):
    total = 0.0
    for q, r in zip(charges_c, positions):
        dist = np.linalg.norm(point - r)
        if dist < 1e-12:
            continue
        total += K * q / dist
    return total


def make_force_table(labels, charges_c, charges_uC, positions):
    rows = []
    for i, label in enumerate(labels):
        force = net_force_on_charge(charges_c, positions, i)
        rows.append({
            "label": label,
            "q_uC": charges_uC[i],
            "x_m": positions[i][0],
            "y_m": positions[i][1],
            "Fx_N": force[0],
            "Fy_N": force[1],
            "F_resultan_N": float(np.linalg.norm(force)),
        })
    return pd.DataFrame(rows)


def make_configuration_plot(labels, charges_uC, positions, forces, show_force=True):
    fig = go.Figure()
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    texts = [f"{lab}<br>{q:.2f} μC" for lab, q in zip(labels, charges_uC)]
    sizes = [18 + min(20, abs(q)) for q in charges_uC]
    symbols = ["circle" if q >= 0 else "x" for q in charges_uC]

    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=texts,
        textposition="top center",
        marker=dict(size=sizes, symbol=symbols, line=dict(width=1)),
        name="Muatan"
    ))

    if show_force:
        for i, force in enumerate(forces):
            mag = np.linalg.norm(force)
            if mag < 1e-15:
                continue
            scale = 0.18 / (1 + np.log10(1 + mag))
            dx = force[0] * scale
            dy = force[1] * scale
            fig.add_annotation(
                x=positions[i][0] + dx,
                y=positions[i][1] + dy,
                ax=positions[i][0],
                ay=positions[i][1],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=2,
            )

    fig.update_layout(
        title="Konfigurasi Muatan 2D",
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        yaxis_scaleanchor="x",
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_field_map(charges_c, charges_uC, positions, labels, xlim=(-2, 2), ylim=(-2, 2), grid_n=19):
    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    P = np.zeros_like(X)

    for i in range(grid_n):
        for j in range(grid_n):
            point = np.array([X[i, j], Y[i, j]])
            field = electric_field_at_point(point, charges_c, positions)
            potential = electric_potential_at_point(point, charges_c, positions)
            field_mag = np.linalg.norm(field)
            if field_mag > 1e-12:
                U[i, j] = field[0] / field_mag
                V[i, j] = field[1] / field_mag
            P[i, j] = potential

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xs,
        y=ys,
        z=P,
        contours=dict(showlabels=False),
        showscale=True,
        opacity=0.7,
        colorbar=dict(title="V (Volt)"),
        name="Potensial"
    ))
    fig.add_trace(go.Cone(
        x=X.flatten(),
        y=Y.flatten(),
        z=np.zeros(X.size),
        u=U.flatten(),
        v=V.flatten(),
        w=np.zeros(X.size),
        sizemode="absolute",
        sizeref=0.25,
        anchor="tail",
        showscale=False,
        name="Medan"
    ))

    fig.add_trace(go.Scatter(
        x=[p[0] for p in positions],
        y=[p[1] for p in positions],
        mode="markers+text",
        text=[f"{lab}<br>{q:.1f} μC" for lab, q in zip(labels, charges_uC)],
        textposition="top center",
        marker=dict(size=14, symbol=["circle" if q >= 0 else "x" for q in charges_uC]),
        name="Muatan"
    ))

    fig.update_layout(
        title="Peta Potensial dan Arah Medan",
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis=dict(visible=False),
            camera_eye=dict(x=0.0, y=0.0, z=2.2),
            aspectmode="data",
        ),
        height=700,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def add_experiment_snapshot(df_force):
    st.session_state.exp_data = pd.concat([
        st.session_state.exp_data,
        df_force
    ], ignore_index=True)


init_state()

st.title("Coulomb Studio 2D")
st.caption("Aplikasi pembelajaran fisika untuk mahasiswa sarjana. Fokus pada hukum Coulomb, superposisi, medan listrik, dan potensial listrik.")

with st.sidebar:
    st.header("Pengaturan Sistem")
    n_charges = st.slider("Jumlah muatan", 2, 8, 3)
    st.write("Muatan positif ditandai lingkaran. Muatan negatif ditandai silang.")
    probe_x = st.slider("x titik uji (m)", -2.0, 2.0, 0.5, 0.1)
    probe_y = st.slider("y titik uji (m)", -2.0, 2.0, 0.5, 0.1)
    grid_n = st.slider("Kerapatan grid peta", 11, 25, 17, 2)
    show_force = st.checkbox("Tampilkan vektor gaya resultan", value=True)

labels = []
charges_uC = []
positions = []

st.subheader("Input muatan")
input_cols = st.columns(2)
for i in range(n_charges):
    with input_cols[i % 2]:
        st.markdown(f"Muatan {i+1}")
        q = st.slider(f"q{i+1} (μC)", -20.0, 20.0, float(5 - i * 3), 0.5, key=f"q_{i}")
        x = st.slider(f"x{i+1} (m)", -2.0, 2.0, float(-1.2 + i * 0.9), 0.1, key=f"x_{i}")
        y = st.slider(f"y{i+1} (m)", -2.0, 2.0, float(-0.6 + i * 0.6), 0.1, key=f"y_{i}")
        labels.append(f"q{i+1}")
        charges_uC.append(q)
        positions.append(np.array([x, y]))

charges_c = np.array(charges_uC) * MICRO
probe_point = np.array([probe_x, probe_y])
force_vectors = [net_force_on_charge(charges_c, positions, i) for i in range(n_charges)]
force_df = make_force_table(labels, charges_c, charges_uC, positions)
field_probe = electric_field_at_point(probe_point, charges_c, positions)
potential_probe = electric_potential_at_point(probe_point, charges_c, positions)
field_probe_mag = np.linalg.norm(field_probe)

st.latex(r"\vec{F}_{ij} = k\frac{q_i q_j}{r_{ij}^3}\vec{r}_{ij}")
st.latex(r"\vec{E}(\vec{r}) = \sum_i k\frac{q_i}{|\vec{r}-\vec{r}_i|^3}(\vec{r}-\vec{r}_i)")

summary_cols = st.columns(4)
summary_cols[0].metric("Jumlah muatan", n_charges)
summary_cols[1].metric("|E| di titik uji", f"{field_probe_mag:.3e} N/C")
summary_cols[2].metric("V di titik uji", f"{potential_probe:.3e} V")
summary_cols[3].metric("Titik uji", f"({probe_x:.1f}, {probe_y:.1f}) m")


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Simulasi", "Praktikum Virtual", "Kuis", "Teori Singkat", "Panduan GitHub"
])

with tab1:
    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        fig_config = make_configuration_plot(labels, charges_uC, positions, force_vectors, show_force=show_force)
        st.plotly_chart(fig_config, use_container_width=True)
    with col_b:
        st.subheader("Gaya resultan pada tiap muatan")
        st.dataframe(force_df, use_container_width=True, hide_index=True)
        st.write("Arah panah menunjukkan arah gaya resultan yang bekerja pada masing-masing muatan akibat semua muatan lain.")
        st.subheader("Medan dan potensial di titik uji")
        st.write(f"Ex = {field_probe[0]:.3e} N/C")
        st.write(f"Ey = {field_probe[1]:.3e} N/C")
        st.write(f"V = {potential_probe:.3e} V")

    st.subheader("Peta potensial dan arah medan")
    st.plotly_chart(
        make_field_map(charges_c, charges_uC, positions, labels, grid_n=grid_n),
        use_container_width=True,
    )

with tab2:
    st.subheader("Praktikum virtual")
    st.write("Gunakan bagian ini untuk mengumpulkan data. Ubah konfigurasi muatan, lalu simpan hasil gaya resultan untuk dianalisis.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Simpan snapshot eksperimen"):
            add_experiment_snapshot(force_df)
            st.success("Snapshot disimpan ke tabel eksperimen.")
    with c2:
        if st.button("Hapus semua data eksperimen"):
            st.session_state.exp_data = st.session_state.exp_data.iloc[0:0]
            st.warning("Data eksperimen dihapus.")

    st.dataframe(st.session_state.exp_data, use_container_width=True, hide_index=True)

    if not st.session_state.exp_data.empty:
        df = st.session_state.exp_data.copy()
        grouped = df.groupby("label", as_index=False)["F_resultan_N"].mean()
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=grouped["label"], y=grouped["F_resultan_N"], name="Rata-rata |F|"))
        fig_bar.update_layout(title="Rata-rata gaya resultan per muatan", xaxis_title="Muatan", yaxis_title="|F| (N)", height=420)
        st.plotly_chart(fig_bar, use_container_width=True)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Unduh data eksperimen CSV",
            data=csv_data,
            file_name="data_praktikum_coulomb_2d.csv",
            mime="text/csv",
        )

        st.subheader("Pertanyaan analisis")
        st.write("1. Muatan mana yang menerima gaya resultan paling besar. Jelaskan berdasarkan posisi dan besar muatan lain.")
        st.write("2. Bandingkan konfigurasi simetris dan tidak simetris. Kapan resultan gaya dapat mendekati nol.")
        st.write("3. Ubah satu muatan menjadi negatif. Amati perubahan pola arah gaya dan bentuk equipotensial.")

with tab3:
    st.subheader("Kuis konsep")
    q1 = st.radio(
        "Jika semua jarak antar muatan diperbesar dua kali, besar gaya Coulomb antar pasangan muatan menjadi...",
        ["tetap", "dua kali", "empat kali", "seperempat kali"],
        key="quiz1",
    )
    q2 = st.radio(
        "Potensial listrik adalah besaran...",
        ["vektor", "skalar", "selalu nol", "selalu positif"],
        key="quiz2",
    )
    q3 = st.radio(
        "Pada titik yang sangat dekat dengan muatan positif tunggal, arah medan listrik...",
        ["menuju muatan", "menjauhi muatan", "selalu nol", "acak"],
        key="quiz3",
    )
    q4 = st.radio(
        "Pada sistem banyak muatan, resultan medan dihitung dengan...",
        ["menjumlahkan potensial saja", "menjumlahkan vektor medan dari tiap muatan", "mengalikan semua medan", "mengambil medan terbesar saja"],
        key="quiz4",
    )

    if st.button("Nilai kuis"):
        score = 0
        score += int(q1 == "seperempat kali")
        score += int(q2 == "skalar")
        score += int(q3 == "menjauhi muatan")
        score += int(q4 == "menjumlahkan vektor medan dari tiap muatan")
        st.session_state.quiz_submitted = True
        st.metric("Skor", f"{score}/4")
        if score < 4:
            st.write("Tinjau kembali konsep hubungan gaya dengan jarak, sifat skalar potensial, arah medan, dan prinsip superposisi.")
        else:
            st.success("Semua jawaban benar.")

with tab4:
    st.subheader("Teori singkat")
    st.write("Hukum Coulomb menyatakan bahwa besar gaya listrik antara dua muatan titik berbanding lurus dengan hasil kali muatan dan berbanding terbalik dengan kuadrat jarak. Pada sistem banyak muatan, gaya total dan medan total diperoleh dengan prinsip superposisi.")
    st.write("Medan listrik adalah besaran vektor. Karena itu arah sangat penting. Potensial listrik adalah besaran skalar. Karena itu potensial dari banyak muatan cukup dijumlahkan secara aljabar.")
    st.write("Dalam konfigurasi simetris, sebagian komponen gaya atau medan dapat saling meniadakan. Inilah alasan mengapa analisis komponen x dan y sangat penting pada sistem dua dimensi.")
    st.write("Untuk mahasiswa sarjana, fokus utama bukan hanya menghitung besar gaya, tetapi juga membaca pola superposisi, menghubungkan geometri dengan arah vektor, dan menafsirkan garis equipotensial serta perubahan energi potensial.")

with tab5:
    st.subheader("Struktur repo GitHub")
    st.code(
        """simulasi-hukum-coulomb/\n├── app.py\n├── requirements.txt\n├── README.md\n└── .gitignore""",
        language="text"
    )
    st.subheader("Perintah Git dasar")
    st.code(
        """git init\ngit add .\ngit commit -m \"Initial commit Coulomb Studio 2D\"\ngit branch -M main\ngit remote add origin https://github.com/USERNAME/NAMA-REPO.git\ngit push -u origin main""",
        language="bash"
    )
    st.subheader("Perintah menjalankan lokal")
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
