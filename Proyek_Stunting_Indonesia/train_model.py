from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Data Stunting Indonesia.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model_stunting.joblib"
DATA_URL = "https://raw.githubusercontent.com/samuelsebastiann11-commits/Sistem-Cerdas/refs/heads/master/Data%20Stunting%20Indonesia.csv"

def load_and_prepare_data():
    try:
        df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8-sig")
    except FileNotFoundError:
        df = pd.read_csv(DATA_URL, sep=";", encoding="utf-8-sig")

    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    if "Provinsi" not in df.columns:
        raise ValueError(f"Kolom Provinsi tidak ditemukan. Kolom tersedia: {df.columns.tolist()}")

    year_columns = [col for col in df.columns if str(col).isdigit()]
    if not year_columns:
        raise ValueError("Kolom tahun tidak ditemukan.")

    df["Provinsi"] = df["Provinsi"].astype(str).str.strip().str.upper()
    for col in year_columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce"
        )

    df = df.dropna(subset=["Provinsi"])
    long_df = df.melt(
        id_vars="Provinsi",
        value_vars=year_columns,
        var_name="Tahun",
        value_name="Nilai"
    )
    long_df["Tahun"] = pd.to_numeric(long_df["Tahun"], errors="coerce")
    long_df["Nilai"] = pd.to_numeric(long_df["Nilai"], errors="coerce")
    long_df = long_df.dropna(subset=["Tahun", "Nilai"])
    long_df["Tahun"] = long_df["Tahun"].astype(int)
    return long_df.sort_values(["Provinsi", "Tahun"]).reset_index(drop=True)

def main():
    data = load_and_prepare_data()
    years = sorted(data["Tahun"].unique())
    if len(years) < 3:
        raise ValueError("Data minimal harus memiliki tiga tahun.")

    last_year = years[-1]
    train_data = data[data["Tahun"] < last_year]
    test_data = data[data["Tahun"] == last_year]

    validation_models = {}
    actual, predicted = [], []
    for province, group in train_data.groupby("Provinsi"):
        if len(group) < 2:
            continue
        model = LinearRegression()
        model.fit(group[["Tahun"]], group["Nilai"])
        validation_models[province] = model

    for _, row in test_data.iterrows():
        model = validation_models.get(row["Provinsi"])
        if model is not None:
            pred = float(model.predict(pd.DataFrame({"Tahun": [row["Tahun"]]}))[0])
            actual.append(float(row["Nilai"]))
            predicted.append(pred)

    mae = mean_absolute_error(actual, predicted) if actual else None
    print(f"Evaluasi tahun {last_year}:")
    if mae is not None:
        print(f"MAE: {mae:.3f} poin")
        if len(actual) > 1:
            print(f"R²: {r2_score(actual, predicted):.3f}")

    final_models = {}
    for province, group in data.groupby("Provinsi"):
        model = LinearRegression()
        model.fit(group[["Tahun"]], group["Nilai"])
        final_models[province] = model

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "models": final_models,
        "years": years,
        "provinces": sorted(final_models.keys()),
        "mae_last_year": mae,
        "last_year": last_year
    }, MODEL_PATH)

    print(f"Jumlah provinsi: {len(final_models)}")
    print(f"Jumlah data bersih: {len(data)}")
    print(f"Model disimpan: {MODEL_PATH}")

if __name__ == "__main__":
    main()