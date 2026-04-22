import io
import json
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Coulomb Studio 2D v4", page_icon="⚡", layout="wide")

K = 8.9875517923e9
MICRO = 1e-6


def init_state():
    defaults = {
        "exp_data": pd.DataFrame(columns=[
            "student_name", "student_id", "class_name", "snapshot_id", "label", "q_uC",
            "x_m", "y_m", "Fx_N", "Fy_N", "F_resultan_N", "E_probe_NC", "V_probe_V"
        ]),
        "ct_log": pd.DataFrame(columns=[
            "student_name", "student_id", "class_name", "decomposition",
            "pattern_recognition", "abstraction", "algorithm_design", "debugging"
        ]),
        "quiz_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


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


def make_force_table(labels, charges_uC, charges_c, positions, e_probe_mag, v_probe, student_name, student_id, class_name, snapshot_id):
    rows = []
    for i, label in enumerate(labels):
        force = net_force_on_charge(charges_c, positions, i)
        rows.append({
            "student_name": student_name,
            "student_id": student_id,
            "class_name": class_name,
            "snapshot_id": snapshot_id,
            "label": label,
            "q_uC": charges_uC[i],
            "x_m": positions[i][0],
            "y_m": positions[i][1],
            "Fx_N": force[0],
            "Fy_N": force[1],
            "F_resultan_N": float(np.linalg.norm(force)),
            "E_probe_NC": e_probe_mag,
            "V_probe_V": v_probe,
        })
    return pd.DataFrame(rows)


def make_configuration_plot(labels, charges_uC, positions, forces, show_force=True):
    fig = go.Figure()
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    texts = [f"{lab}<br>{q:.2f} μC" for lab, q in zip(labels, charges_uC)]
    sizes = [18 + min(18, abs(q)) for q in charges_uC]
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
            scale = 0.22 / (1 + np.log10(1 + mag))
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
        title="Konfigurasi Muatan 2D dan Vektor Gaya Resultan",
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        yaxis_scaleanchor="x",
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_field_and_potential_map(charges_c, charges_uC, positions, labels, xlim=(-2, 2), ylim=(-2, 2), grid_n=21):
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
            mag = np.linalg.norm(field)
            if mag > 1e-12:
                U[i, j] = field[0] / mag
                V[i, j] = field[1] / mag
            P[i, j] = potential

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xs,
        y=ys,
        z=P,
        colorscale="RdBu",
        contours=dict(showlabels=False),
        colorbar=dict(title="V (Volt)"),
        opacity=0.78,
        name="Potensial"
    ))
    fig.add_trace(go.Scatter(
        x=X.flatten(),
        y=Y.flatten(),
        mode="markers",
        marker=dict(size=1, opacity=0),
        hoverinfo="skip",
        showlegend=False
    ))
    step = max(1, grid_n // 12)
    for i in range(0, grid_n, step):
        for j in range(0, grid_n, step):
            fig.add_annotation(
                x=X[i, j] + 0.12 * U[i, j],
                y=Y[i, j] + 0.12 * V[i, j],
                ax=X[i, j],
                ay=Y[i, j],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=0.8,
                arrowwidth=1,
                opacity=0.7,
            )

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
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        yaxis_scaleanchor="x",
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def dataframe_to_csv_download(df, file_name):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Unduh CSV", data=csv, file_name=file_name, mime="text/csv")


def push_to_google_sheet(df_rows, worksheet_name="CoulombSnapshots"):
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        service_account_info = dict(st.secrets["gcp_service_account"])
        spreadsheet_id = st.secrets["google_sheet_id"]
        creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
        client = gspread.authorize(creds)
        sh = client.open_by_key(spreadsheet_id)
        try:
            ws = sh.worksheet(worksheet_name)
        except Exception:
            ws = sh.add_worksheet(title=worksheet_name, rows=1000, cols=30)
            ws.append_row(list(df_rows.columns))

        existing_headers = ws.row_values(1)
        if not existing_headers:
            ws.append_row(list(df_rows.columns))
        elif existing_headers != list(df_rows.columns):
            st.warning("Header sheet tidak sama dengan data saat ini. Buat worksheet baru atau sesuaikan header.")
            return False, "Header worksheet tidak sesuai"

        ws.append_rows(df_rows.astype(str).values.tolist())
        return True, f"Data berhasil dikirim ke worksheet {worksheet_name}."
    except Exception as e:
        return False, str(e)


init_state()

st.title("Coulomb Studio 2D v4")
st.caption("Media pembelajaran fisika level sarjana untuk hukum Coulomb, superposisi, medan listrik, potensial listrik, dan latihan computational thinking.")

with st.sidebar:
    st.header("Identitas mahasiswa")
    student_name = st.text_input("Nama")
    student_id = st.text_input("NIM")
    class_name = st.text_input("Kelas")

    st.header("Pengaturan simulasi")
    n_charges = st.slider("Jumlah muatan", 2, 8, 4)
    probe_x = st.slider("x titik uji (m)", -2.0, 2.0, 0.4, 0.1)
    probe_y = st.slider("y titik uji (m)", -2.0, 2.0, -0.2, 0.1)
    grid_n = st.slider("Kerapatan grid peta", 11, 25, 17, 2)
    show_force = st.checkbox("Tampilkan vektor gaya resultan", value=True)
    st.caption("Muatan positif ditandai lingkaran. Muatan negatif ditandai silang.")

labels = []
charges_uC = []
positions = []

st.subheader("Input konfigurasi muatan")
cols = st.columns(2)
for i in range(n_charges):
    with cols[i % 2]:
        st.markdown(f"Muatan {i+1}")
        q = st.slider(f"q{i+1} (μC)", -20.0, 20.0, float(6 - i * 3), 0.5, key=f"q_{i}")
        x = st.slider(f"x{i+1} (m)", -2.0, 2.0, float(-1.2 + i * 0.7), 0.1, key=f"x_{i}")
        y = st.slider(f"y{i+1} (m)", -2.0, 2.0, float(-0.8 + i * 0.4), 0.1, key=f"y_{i}")
        labels.append(f"q{i+1}")
        charges_uC.append(q)
        positions.append(np.array([x, y]))

charges_c = np.array(charges_uC) * MICRO
probe_point = np.array([probe_x, probe_y])
forces = [net_force_on_charge(charges_c, positions, i) for i in range(n_charges)]
field_probe = electric_field_at_point(probe_point, charges_c, positions)
potential_probe = electric_potential_at_point(probe_point, charges_c, positions)
field_mag = float(np.linalg.norm(field_probe))
snapshot_id = f"SNAP-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"
force_df = make_force_table(labels, charges_uC, charges_c, positions, field_mag, potential_probe, student_name, student_id, class_name, snapshot_id)

st.latex(r"\vec{F}_{ij} = k\frac{q_i q_j}{r_{ij}^3}\vec{r}_{ij}")
st.latex(r"\vec{E}(\vec{r}) = \sum_i k\frac{q_i}{|\vec{r}-\vec{r}_i|^3}(\vec{r}-\vec{r}_i)")
st.latex(r"V(\vec{r}) = \sum_i k\frac{q_i}{|\vec{r}-\vec{r}_i|}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Jumlah muatan", n_charges)
m2.metric("|E| di titik uji", f"{field_mag:.3e} N/C")
m3.metric("V di titik uji", f"{potential_probe:.3e} V")
m4.metric("Titik uji", f"({probe_x:.1f}, {probe_y:.1f}) m")

sim_tab, prac_tab, ct_tab, quiz_tab, theory_tab, github_tab = st.tabs([
    "Simulasi", "Praktikum", "Computational Thinking", "Kuis", "Teori", "GitHub"
])

with sim_tab:
    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(make_configuration_plot(labels, charges_uC, positions, forces, show_force), use_container_width=True)
    with right:
        st.plotly_chart(make_field_and_potential_map(charges_c, charges_uC, positions, labels, grid_n=grid_n), use_container_width=True)
    st.subheader("Tabel gaya resultan")
    st.dataframe(force_df[["label", "q_uC", "x_m", "y_m", "Fx_N", "Fy_N", "F_resultan_N"]], use_container_width=True, hide_index=True)

with prac_tab:
    st.subheader("Praktikum virtual")
    st.write("Gunakan tab ini untuk menyimpan snapshot konfigurasi, mengunduh data, atau mengirim data ke Google Sheet.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Simpan snapshot ke tabel lokal"):
            st.session_state.exp_data = pd.concat([st.session_state.exp_data, force_df], ignore_index=True)
            st.success("Snapshot masuk ke tabel lokal.")
        dataframe_to_csv_download(force_df, f"snapshot_{snapshot_id}.csv")
    with c2:
        send_sheet = st.button("Kirim snapshot ke Google Sheet")
        if send_sheet:
            ok, msg = push_to_google_sheet(force_df, worksheet_name="CoulombSnapshots")
            if ok:
                st.success(msg)
            else:
                st.error(f"Gagal mengirim ke Google Sheet: {msg}")
    st.markdown("Riwayat snapshot lokal")
    st.dataframe(st.session_state.exp_data, use_container_width=True, hide_index=True)
    if not st.session_state.exp_data.empty:
        dataframe_to_csv_download(st.session_state.exp_data, "riwayat_snapshot_coulomb.csv")

with ct_tab:
    st.subheader("Latihan computational thinking")
    st.write("Isi jawaban singkat berdasarkan konfigurasi yang sedang aktif.")
    decomposition = st.text_area(
        "Decomposition. Pecah masalah ini menjadi beberapa langkah analisis.",
        placeholder="Contoh: identifikasi tanda muatan, cek posisi relatif, hitung kontribusi gaya dari tiap muatan, jumlahkan vektor...",
        height=120,
    )
    pattern = st.text_area(
        "Pattern recognition. Pola apa yang Anda lihat saat jarak diubah atau tanda muatan diganti?",
        height=120,
    )
    abstraction = st.text_area(
        "Abstraction. Variabel mana yang paling penting dan mana yang dapat diabaikan pada model ini?",
        height=120,
    )
    algorithm = st.text_area(
        "Algorithm design. Tulis algoritma singkat untuk menghitung gaya resultan pada satu muatan.",
        height=120,
    )
    debugging = st.text_area(
        "Debugging. Jika hasil simulasi tidak masuk akal, apa yang akan Anda periksa lebih dulu?",
        height=120,
    )

    if st.button("Simpan jawaban CT"):
        row = pd.DataFrame([{
            "student_name": student_name,
            "student_id": student_id,
            "class_name": class_name,
            "decomposition": decomposition,
            "pattern_recognition": pattern,
            "abstraction": abstraction,
            "algorithm_design": algorithm,
            "debugging": debugging,
        }])
        st.session_state.ct_log = pd.concat([st.session_state.ct_log, row], ignore_index=True)
        st.success("Jawaban CT tersimpan di tabel lokal.")

    st.dataframe(st.session_state.ct_log, use_container_width=True, hide_index=True)
    if not st.session_state.ct_log.empty:
        dataframe_to_csv_download(st.session_state.ct_log, "jawaban_ct_coulomb.csv")
        if st.button("Kirim jawaban CT ke Google Sheet"):
            ok, msg = push_to_google_sheet(st.session_state.ct_log.tail(1), worksheet_name="CoulombCT")
            if ok:
                st.success(msg)
            else:
                st.error(f"Gagal mengirim jawaban CT: {msg}")

with quiz_tab:
    st.subheader("Kuis konsep singkat")
    q1 = st.radio(
        "Jika dua muatan bertanda sama didekatkan, apa yang terjadi pada besar gaya Coulomb?",
        ["Berkurang", "Tetap", "Meningkat"],
        index=None,
    )
    q2 = st.radio(
        "Jika jarak menjadi dua kali semula, besar gaya menjadi...",
        ["2 kali", "1/2 kali", "1/4 kali"],
        index=None,
    )
    q3 = st.radio(
        "Potensial listrik adalah besaran...",
        ["Skalar", "Vektor", "Tidak punya satuan"],
        index=None,
    )
    if st.button("Periksa jawaban kuis"):
        score = 0
        score += int(q1 == "Meningkat")
        score += int(q2 == "1/4 kali")
        score += int(q3 == "Skalar")
        st.info(f"Skor Anda: {score}/3")
        if score < 3:
            st.write("Periksa lagi hubungan F ∝ 1/r² dan perbedaan antara medan listrik dengan potensial listrik.")
        else:
            st.success("Semua benar.")

with theory_tab:
    st.subheader("Ringkasan teori")
    st.write(
        "Hukum Coulomb menyatakan bahwa besar gaya listrik antara dua muatan titik sebanding dengan hasil kali kedua muatan dan berbanding terbalik dengan kuadrat jarak. "
        "Pada sistem multi-muatan, gaya dan medan dihitung dengan prinsip superposisi. Karena gaya adalah besaran vektor, kontribusi dari setiap muatan harus dijumlahkan secara komponen. "
        "Potensial listrik berbeda dari medan listrik karena potensial adalah besaran skalar. Oleh sebab itu, penjumlahan potensial lebih sederhana daripada penjumlahan medan."
    )
    st.write(
        "Pada level sarjana, mahasiswa perlu membaca konfigurasi geometris, memprediksi arah gaya, membedakan tarikan dan tolakan, dan menafsirkan peta potensial serta arah medan. "
        "Aplikasi ini sengaja menampilkan multi-muatan agar konsep superposisi terlihat jelas."
    )

with github_tab:
    st.subheader("Struktur repository")
    st.code(
        """simulasi-hukum-coulomb/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore"""
    )
    st.subheader("Perintah dasar GitHub")
    st.code(
        """git init
git add .
git commit -m \"Initial commit Coulomb Studio 2D v4\"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main""",
        language="bash"
    )
    st.subheader("Catatan Google Sheet")
    st.write("Jika ingin mengirim data ke Google Sheet dari Streamlit Cloud, tambahkan secrets berikut.")
    st.code(
        """google_sheet_id = "YOUR_SHEET_ID"

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
client_email = "..."
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
universe_domain = "googleapis.com""" ,
        language="toml"
    )
