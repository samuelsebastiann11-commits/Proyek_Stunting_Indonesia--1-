from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Data Stunting Indonesia.csv"
MODEL_PATH = BASE_DIR / "models" / "model_stunting.joblib"
DATA_URL = "https://raw.githubusercontent.com/samuelsebastiann11-commits/Sistem-Cerdas/refs/heads/master/Data%20Stunting%20Indonesia.csv"

st.set_page_config(page_title="Monitoring Data Stunting Indonesia", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    try:
        raw = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8-sig")
    except FileNotFoundError:
        raw = pd.read_csv(DATA_URL, sep=";", encoding="utf-8-sig")

    raw.columns = [str(col).replace("\ufeff", "").strip() for col in raw.columns]
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]
    raw["Provinsi"] = raw["Provinsi"].astype(str).str.strip().str.upper()
    year_cols = [col for col in raw.columns if str(col).isdigit()]
    for col in year_cols:
        raw[col] = pd.to_numeric(
            raw[col].astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce"
        )
    long_data = raw.melt(
        id_vars="Provinsi",
        value_vars=year_cols,
        var_name="Tahun",
        value_name="Nilai"
    )
    long_data["Tahun"] = pd.to_numeric(long_data["Tahun"], errors="coerce")
    long_data["Nilai"] = pd.to_numeric(long_data["Nilai"], errors="coerce")
    return long_data.dropna().sort_values(["Provinsi", "Tahun"])

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model belum ditemukan. Jalankan `python train_model.py` terlebih dahulu.")
        st.stop()
    return joblib.load(MODEL_PATH)

data = load_data()
bundle = load_model()
models = bundle["models"]
years = bundle["years"]
latest_year = max(years)
provinces = sorted([p for p in data["Provinsi"].unique() if p in models])
latest = data[data["Tahun"] == latest_year]

st.title("📊 Dashboard Monitoring dan Estimasi Tren Stunting")
st.write("Lihat data historis setiap provinsi dan estimasi tren menggunakan Linear Regression.")
st.info("Prediksi merupakan estimasi berdasarkan pola historis dataset, bukan angka resmi. Data hanya mencakup empat tahun sehingga hasil perlu ditafsirkan secara hati-hati.")

c1, c2, c3 = st.columns(3)
c1.metric("Jumlah provinsi", data["Provinsi"].nunique())
c2.metric("Tahun terakhir dataset", latest_year)
c3.metric("Rata-rata nilai tahun terakhir", f"{latest['Nilai'].mean():.2f}")

st.divider()
left, right = st.columns([1, 1])

with left:
    st.subheader("Prediksi berdasarkan provinsi")
    selected_province = st.selectbox("Pilih provinsi", provinces)
    target_year = st.number_input(
        "Tahun yang ingin diestimasi",
        min_value=int(latest_year + 1),
        max_value=int(latest_year + 3),
        value=int(latest_year + 1),
        step=1
    )
    history = data[data["Provinsi"] == selected_province].sort_values("Tahun")
    model = models[selected_province]
    estimate = float(model.predict(pd.DataFrame({"Tahun": [int(target_year)]}))[0])
    estimate = max(0.0, estimate)
    st.metric(f"Estimasi nilai tahun {int(target_year)}", f"{estimate:.2f}")
    st.caption("Nilai prediksi dibatasi minimal 0. Periksa kembali satuan dan definisi nilai pada sumber dataset.")

with right:
    st.subheader(f"Grafik tren {selected_province}")
    chart = history[["Tahun", "Nilai"]].copy()
    chart.loc[len(chart)] = [int(target_year), estimate]
    chart = chart.set_index("Tahun").sort_index()
    st.line_chart(chart["Nilai"], use_container_width=True)
    st.caption("Grafik menampilkan data historis dan estimasi pada tahun pilihan.")

st.divider()
st.subheader(f"10 provinsi dengan nilai tertinggi pada tahun {latest_year}")
ranking = latest.sort_values("Nilai", ascending=False).head(10).set_index("Provinsi")
st.bar_chart(ranking["Nilai"], use_container_width=True)

with st.expander("Lihat tabel data yang digunakan"):
    table = data.pivot(index="Provinsi", columns="Tahun", values="Nilai").reset_index()
    st.dataframe(table, use_container_width=True)

with st.expander("Informasi model dan evaluasi"):
    st.write("Metode: Linear Regression, dibuat terpisah untuk setiap provinsi.")
    st.write(f"Evaluasi dilakukan dengan menguji tahun {bundle['last_year']} menggunakan tahun-tahun sebelumnya.")
    if bundle["mae_last_year"] is not None:
        st.write(f"MAE: {bundle['mae_last_year']:.3f} poin.")