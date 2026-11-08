from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

CSV_PATH = Path(__file__).parent / "data" / "hajj_umrah_dataset.csv"
ORDER = ["Low", "Medium", "High"]  # urutan tampil


def load_data():
    return pd.read_csv(CSV_PATH)


@app.route("/")
def index():
    return render_template("index.html", levels=ORDER)


@app.route("/api/data")
def api_data():
    """Kirim data + ringkasan. Bisa difilter: /api/data?density=High"""
    df = load_data()
    density = request.args.get("density", "All")
    if density in ORDER:
        df = df[df["Crowd_Density"] == density]

    # ringkasan per tingkat kepadatan
    group = (
        df.groupby("Crowd_Density")
        .agg(
            jumlah=("Movement_Speed", "size"),
            kecepatan=("Movement_Speed", "mean"),
            jarak=("Distance_Between_People_m", "mean"),
        )
        .round(2)
        .reindex([o for o in ORDER if o in df["Crowd_Density"].unique()])
    )

    # histogram kecepatan (bin 0.2 m/s)
    bins = [round(0.2 * i, 1) for i in range(0, 9)]  # 0.0 .. 1.6
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    hist = pd.cut(df["Movement_Speed"], bins=bins, labels=labels).value_counts().reindex(labels)

    corr = df["Movement_Speed"].corr(df["Distance_Between_People_m"])

    return jsonify(
        {
            "total": int(len(df)),
            "avg_speed": round(df["Movement_Speed"].mean(), 2),
            "avg_distance": round(df["Distance_Between_People_m"].mean(), 2),
            "corr": None if pd.isna(corr) else round(float(corr), 2),
            "group": {
                "labels": group.index.tolist(),
                "jumlah": group["jumlah"].tolist(),
                "kecepatan": group["kecepatan"].tolist(),
                "jarak": group["jarak"].tolist(),
            },
            "hist": {"labels": labels, "values": hist.fillna(0).astype(int).tolist()},
            "scatter": {
                level: df[df["Crowd_Density"] == level][
                    ["Distance_Between_People_m", "Movement_Speed"]
                ]
                .rename(columns={"Distance_Between_People_m": "x", "Movement_Speed": "y"})
                .to_dict("records")
                for level in ORDER
            },
            "rows": df.to_dict("records"),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)