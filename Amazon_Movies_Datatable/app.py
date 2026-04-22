from flask import Flask, render_template
import pandas as pd
from scraper import run_notebook, CSV_OUT

app = Flask(__name__)

# ── Run notebook only if we have no CSV yet ─────────────────────
if CSV_OUT.exists():
    print("📄 Loading existing CSV …")
    df = pd.read_csv(CSV_OUT)
else:
    print("🚀 No CSV found – running notebook once …")
    df = run_notebook()
    # run_notebook either returned a DataFrame or the notebook saved the CSV
    if df is None and CSV_OUT.exists():
        df = pd.read_csv(CSV_OUT)
    elif df is None:
        raise RuntimeError("Notebook produced neither final_df nor CSV")

movies_dict = df.to_dict(orient="records")
columns     = df.columns
# ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", movies=movies_dict, columns=columns)

if __name__ == "__main__":
    # debug=True shows errors but `use_reloader=False` prevents double‑run
    app.run(debug=True, use_reloader=False)
