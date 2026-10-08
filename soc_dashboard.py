"""
NIDS SOC Dashboard (Streamlit)
------------------------------
Run:  streamlit run soc_dashboard.py

Reads the artifacts produced by your pipeline:
  prediction_confidence.csv / calibrated_confidence_results.csv
  adaptive_threshold_results.csv, misclassified_predictions.csv
  model_comparison.csv, feature_importance.csv
  final_xgboost.pkl + confidence_calibrator.pkl   (only for the Live Scoring tab)

Optional: class_names.json  ->  {"0": "Benign", "1": "DDoS", ...}
(Labels were produced by LabelEncoder, i.e. alphabetical order of the original
"Label" column. Use `dict(enumerate(encoder.classes_))` to export the mapping.)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="NIDS SOC Dashboard", page_icon="🛡️", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem;}
    div[data-testid="stMetric"] {
        background: #121a2b; border: 1px solid #22304d; border-left: 4px solid #3b82f6;
        padding: 12px 16px; border-radius: 8px;
    }
    div[data-testid="stMetricLabel"] {color: #8da2c4;}
    h1, h2, h3 {letter-spacing: .2px;}
    </style>
    """,
    unsafe_allow_html=True,
)
PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
GOOD, BAD, WARN, INFO = "#22c55e", "#ef4444", "#f59e0b", "#3b82f6"

# ------------------------------------------------------------------ data loading
st.sidebar.title("🛡️ NIDS SOC")
data_dir = Path(st.sidebar.text_input("Data folder", str(Path(__file__).parent)))


@st.cache_data(show_spinner="Loading predictions…")
def load_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_predictions(folder: str) -> pd.DataFrame | None:
    for name in ("prediction_confidence.csv", "calibrated_confidence_results.csv"):
        df = load_csv(str(Path(folder) / name))
        if df is not None:
            df = df.copy()
            df["Correct"] = df["Correct"].astype(bool)
            df["Actual"] = df["Actual"].astype(int)
            df["Predicted"] = df["Predicted"].astype(int)
            df["Alert_ID"] = np.arange(len(df))
            return df
    return None


df = load_predictions(str(data_dir))
if df is None:
    st.error(f"No prediction CSV found in `{data_dir}`. Point the sidebar to your results folder.")
    st.stop()

models_df = load_csv(str(data_dir / "model_comparison.csv"))
fi_df = load_csv(str(data_dir / "feature_importance.csv"))

# class-name mapping
names: dict[int, str] = {}
cn_path = data_dir / "class_names.json"
upload = st.sidebar.file_uploader("Class names (JSON, optional)", type="json")
try:
    raw = json.load(upload) if upload else (json.loads(cn_path.read_text()) if cn_path.exists() else {})
    names = {int(k): v for k, v in raw.items()}
except Exception as e:  # noqa: BLE001
    st.sidebar.warning(f"Could not read class names: {e}")


def cname(c: int) -> str:
    return names.get(int(c), f"Class {int(c)}")


df["Actual_Name"] = df["Actual"].map(cname)
df["Predicted_Name"] = df["Predicted"].map(cname)

# Benign class: class 0 by default (it's ~78% of your sample). Changeable.
benign_id = st.sidebar.number_input("Benign class ID", 0, int(df["Actual"].max()), 0)

st.sidebar.divider()
threshold = st.sidebar.slider(
    "Auto-alert threshold (calibrated confidence)", 0.50, 0.999, 0.99, 0.001, format="%.3f",
    help="Predictions at or above this are auto-actioned; the rest go to an analyst.",
)
df["Action"] = np.where(df["Calibrated_Confidence"] >= threshold, "Automatic Alert", "Analyst Review")
auto = df[df["Action"] == "Automatic Alert"]
queue = df[df["Action"] == "Analyst Review"]
wrong = df[~df["Correct"]]

# ------------------------------------------------------------------ helpers
def ece(conf: pd.Series, correct: pd.Series, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return total


def reliability(conf_col: str, bins: int = 12) -> pd.DataFrame:
    d = df[[conf_col, "Correct"]].copy()
    d["bin"] = pd.cut(d[conf_col], np.linspace(0, 1, bins + 1), include_lowest=True)
    g = d.groupby("bin", observed=True).agg(conf=(conf_col, "mean"), acc=("Correct", "mean"), n=("Correct", "size"))
    return g.reset_index(drop=True)


def sweep(thresholds) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        a = df["Calibrated_Confidence"] >= t
        wa = (~df["Correct"] & a).sum()
        rows.append({
            "Threshold": t,
            "Auto Coverage (%)": a.mean() * 100,
            "Auto Accuracy (%)": df.loc[a, "Correct"].mean() * 100 if a.any() else np.nan,
            "Wrong Auto Alerts": int(wa),
            "Analyst Reviews": int((~a).sum()),
            "Errors Caught (%)": (1 - wa / max(len(wrong), 1)) * 100,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ header
st.title("Network Intrusion Detection — SOC Console")
st.caption(
    f"{len(df):,} scored flows · {df['Actual'].nunique()} classes · XGBoost + logistic confidence calibrator · "
    f"operating threshold **{threshold:.3f}**"
)

tabs = st.tabs(["📊 Overview", "🎚️ Triage & Threshold", "📋 Analyst Queue", "🔍 Error Analysis",
                "🧠 Models & Features", "⚡ Live Scoring"])

# ================================================================== OVERVIEW
with tabs[0]:
    c = st.columns(6)
    c[0].metric("Model accuracy", f"{df['Correct'].mean() * 100:.2f}%")
    c[1].metric("Auto-alert coverage", f"{len(auto) / len(df) * 100:.1f}%")
    c[2].metric("Auto-alert accuracy", f"{auto['Correct'].mean() * 100:.2f}%" if len(auto) else "–")
    c[3].metric("Analyst queue", f"{len(queue):,}", f"{len(queue) / len(df) * 100:.1f}% of flows", delta_color="off")
    c[4].metric("Errors routed to analyst", f"{queue['Correct'].eq(False).sum() / max(len(wrong), 1) * 100:.1f}%")
    c[5].metric("Wrong auto-alerts", f"{int((~auto['Correct']).sum()):,}", delta_color="off")

    left, right = st.columns(2)

    with left:
        st.subheader("Confidence distribution")
        h = df.assign(Outcome=np.where(df["Correct"], "Correct", "Wrong"))
        fig = px.histogram(h, x="Calibrated_Confidence", color="Outcome", nbins=50, barmode="overlay",
                           opacity=0.75, log_y=True, color_discrete_map={"Correct": GOOD, "Wrong": BAD})
        fig.add_vline(x=threshold, line_dash="dash", line_color=WARN, annotation_text="threshold")
        fig.update_layout(**PLOT, height=380, yaxis_title="Flows (log)", xaxis_title="Calibrated confidence")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Reliability: raw vs calibrated")
        r_raw, r_cal = reliability("Raw_Confidence"), reliability("Calibrated_Confidence")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                                 line=dict(dash="dot", color="#64748b")))
        fig.add_trace(go.Scatter(x=r_raw["conf"], y=r_raw["acc"], mode="lines+markers", name="Raw XGBoost",
                                 line=dict(color=WARN)))
        fig.add_trace(go.Scatter(x=r_cal["conf"], y=r_cal["acc"], mode="lines+markers", name="Calibrated",
                                 line=dict(color=INFO)))
        fig.update_layout(**PLOT, height=380, xaxis_title="Mean confidence", yaxis_title="Observed accuracy")
        st.plotly_chart(fig, use_container_width=True)
        e1, e2 = ece(df["Raw_Confidence"], df["Correct"]), ece(df["Calibrated_Confidence"], df["Correct"])
        st.caption(f"Expected calibration error — raw **{e1:.4f}** → calibrated **{e2:.4f}**")

    st.subheader("Traffic mix (ground truth)")
    mix = df["Actual_Name"].value_counts().reset_index()
    mix.columns = ["Class", "Flows"]
    fig = px.bar(mix, x="Class", y="Flows", log_y=True, color_discrete_sequence=[INFO])
    fig.update_layout(**PLOT, height=340, xaxis_tickangle=-60)
    st.plotly_chart(fig, use_container_width=True)

# ================================================================== TRIAGE
with tabs[1]:
    st.subheader("Automation vs. analyst workload")
    st.caption("Move the sidebar slider to see the live operating point on the trade-off curves.")

    sw = sweep(np.round(np.arange(0.50, 0.9995, 0.005), 3))
    cur = sweep([threshold]).iloc[0]

    k = st.columns(4)
    k[0].metric("Auto coverage", f"{cur['Auto Coverage (%)']:.2f}%")
    k[1].metric("Auto accuracy", f"{cur['Auto Accuracy (%)']:.2f}%")
    k[2].metric("Wrong auto-alerts", f"{int(cur['Wrong Auto Alerts']):,}")
    k[3].metric("Errors caught", f"{cur['Errors Caught (%)']:.1f}%")

    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sw["Threshold"], y=sw["Auto Coverage (%)"], name="Auto coverage %",
                                 line=dict(color=INFO)))
        fig.add_trace(go.Scatter(x=sw["Threshold"], y=sw["Errors Caught (%)"], name="Errors caught %",
                                 line=dict(color=GOOD)))
        fig.add_trace(go.Scatter(x=sw["Threshold"], y=sw["Auto Accuracy (%)"], name="Auto accuracy %",
                                 line=dict(color=WARN)))
        fig.add_vline(x=threshold, line_dash="dash", line_color="#94a3b8")
        fig.update_layout(**PLOT, height=380, xaxis_title="Threshold", yaxis_title="%",
                          title="Coverage vs. safety")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.line(sw, x="Analyst Reviews", y="Wrong Auto Alerts", markers=False, title="Cost frontier",
                      color_discrete_sequence=[BAD])
        fig.add_trace(go.Scatter(x=[cur["Analyst Reviews"]], y=[cur["Wrong Auto Alerts"]], mode="markers",
                                 marker=dict(size=14, color=WARN), name="Current"))
        fig.update_layout(**PLOT, height=380, xaxis_title="Analyst reviews (workload)",
                          yaxis_title="Wrong auto-alerts (risk)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Analyst capacity planner")
    p1, p2, p3 = st.columns(3)
    daily = p1.number_input("Flows scored per day", 1000, 100_000_000, 1_000_000, step=50_000)
    mins = p2.number_input("Minutes per manual review", 0.5, 60.0, 5.0, step=0.5)
    hrs = p3.number_input("Analyst hours per shift", 1.0, 12.0, 7.0, step=0.5)
    reviews_day = daily * cur["Analyst Reviews"] / len(df)
    st.info(f"At this threshold ≈ **{reviews_day:,.0f} reviews/day** → "
            f"**{reviews_day * mins / 60 / hrs:,.1f} analyst-shifts/day**; "
            f"≈ **{daily * cur['Wrong Auto Alerts'] / len(df):,.0f} wrong auto-alerts/day** slip through.")

    st.markdown("#### Threshold sweep (your pipeline's selected points)")
    saved = load_csv(str(data_dir / "adaptive_threshold_results.csv"))
    st.dataframe(saved if saved is not None else sw.iloc[::10], use_container_width=True, hide_index=True)

# ================================================================== QUEUE
with tabs[2]:
    st.subheader("Analyst review queue")
    f1, f2, f3 = st.columns(3)
    pred_sel = f1.multiselect("Predicted class", sorted(queue["Predicted_Name"].unique()))
    lo, hi = f2.slider("Calibrated confidence range", 0.0, 1.0, (0.0, float(threshold)), 0.01)
    only_err = f3.checkbox("Only known errors (evaluation view)", value=False,
                           help="Uses ground truth — only meaningful for offline evaluation, not production.")

    q = queue[queue["Calibrated_Confidence"].between(lo, hi)]
    if pred_sel:
        q = q[q["Predicted_Name"].isin(pred_sel)]
    if only_err:
        q = q[~q["Correct"]]
    q = q.sort_values("Calibrated_Confidence")

    st.caption(f"{len(q):,} flows shown · lowest confidence first")
    show = q[["Alert_ID", "Predicted_Name", "Calibrated_Confidence", "Raw_Confidence", "Actual_Name", "Correct"]]
    st.dataframe(
        show.head(2000), use_container_width=True, hide_index=True,
        column_config={
            "Calibrated_Confidence": st.column_config.ProgressColumn("Calibrated conf.", min_value=0, max_value=1,
                                                                    format="%.3f"),
            "Raw_Confidence": st.column_config.NumberColumn("Raw conf.", format="%.3f"),
            "Actual_Name": "Ground truth",
        },
    )
    st.download_button("⬇️ Export queue (CSV)", q.to_csv(index=False).encode(), "analyst_queue.csv", "text/csv")

    st.subheader("What's in the queue?")
    qc = queue["Predicted_Name"].value_counts().head(15).reset_index()
    qc.columns = ["Predicted", "Flows"]
    fig = px.bar(qc, x="Flows", y="Predicted", orientation="h", color_discrete_sequence=[WARN])
    fig.update_layout(**PLOT, height=420, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

# ================================================================== ERRORS
with tabs[3]:
    st.subheader("Where does the model go wrong?")

    # Dangerous misses: real attack predicted as benign
    missed = wrong[(wrong["Predicted"] == benign_id) & (wrong["Actual"] != benign_id)]
    false_alarm = wrong[(wrong["Actual"] == benign_id) & (wrong["Predicted"] != benign_id)]
    blind = wrong[wrong["Calibrated_Confidence"] >= threshold]

    k = st.columns(4)
    k[0].metric("Total errors", f"{len(wrong):,}")
    k[1].metric("Missed attacks (→ benign)", f"{len(missed):,}", help="Attack flows classified as benign — false negatives.")
    k[2].metric("False alarms (benign → attack)", f"{len(false_alarm):,}")
    k[3].metric("Confident blind spots", f"{len(blind):,}", help="Errors above the auto-alert threshold.")

    pairs = (wrong.groupby(["Actual_Name", "Predicted_Name"]).size().reset_index(name="Errors")
             .sort_values("Errors", ascending=False))
    pairs["Pair"] = pairs["Actual_Name"] + "  →  " + pairs["Predicted_Name"]

    a, b = st.columns(2)
    with a:
        st.markdown("**Top confusion pairs (actual → predicted)**")
        fig = px.bar(pairs.head(15), x="Errors", y="Pair", orientation="h", color_discrete_sequence=[BAD])
        fig.update_layout(**PLOT, height=470, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.markdown("**Confusion heatmap (errors only)**")
        top_cls = pd.concat([wrong["Actual_Name"], wrong["Predicted_Name"]]).value_counts().head(12).index
        ct = pd.crosstab(wrong["Actual_Name"], wrong["Predicted_Name"]).reindex(index=top_cls, columns=top_cls,
                                                                                fill_value=0)
        fig = px.imshow(np.log1p(ct.values), x=ct.columns, y=ct.index, color_continuous_scale="Reds",
                        aspect="auto", labels=dict(color="log(1+errors)"))
        fig.update_layout(**PLOT, height=470, xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Per-class precision / recall**")
    cls = pd.DataFrame({
        "Support": df.groupby("Actual_Name").size(),
        "Recall (%)": df.groupby("Actual_Name")["Correct"].mean() * 100,
    })
    pred_n = df.groupby("Predicted_Name").size()
    tp = df[df["Correct"]].groupby("Predicted_Name").size()
    cls["Precision (%)"] = (tp / pred_n * 100).reindex(cls.index)
    cls = cls.sort_values("Recall (%)")
    st.dataframe(cls.style.format({"Recall (%)": "{:.1f}", "Precision (%)": "{:.1f}"})
                 .background_gradient(subset=["Recall (%)", "Precision (%)"], cmap="RdYlGn", vmin=50, vmax=100),
                 use_container_width=True)

    if len(blind):
        st.warning(f"⚠️ {len(blind)} wrong predictions sit above the auto-alert threshold — these bypass analysts. "
                   f"Largest group: **{blind.groupby(['Actual_Name', 'Predicted_Name']).size().idxmax()[0]} → "
                   f"{blind.groupby(['Actual_Name', 'Predicted_Name']).size().idxmax()[1]}**.")
        st.dataframe(blind[["Alert_ID", "Actual_Name", "Predicted_Name", "Calibrated_Confidence"]]
                     .sort_values("Calibrated_Confidence", ascending=False).head(500),
                     use_container_width=True, hide_index=True)

# ================================================================== MODELS
with tabs[4]:
    a, b = st.columns(2)
    with a:
        st.subheader("Model comparison")
        if models_df is not None:
            st.dataframe(models_df.style.format({c: "{:.4f}" for c in models_df.columns[1:5]}
                                                | {"Training Time": "{:.1f}s", "Prediction Time": "{:.3f}s"}),
                         use_container_width=True, hide_index=True)
            m = models_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1 Score"])
            fig = px.bar(m, x="variable", y="value", color="Model", barmode="group")
            fig.update_layout(**PLOT, height=360, xaxis_title="", yaxis_title="", yaxis_range=[0.6, 1.0])
            st.plotly_chart(fig, use_container_width=True)
            st.caption("LightGBM is far below the tree ensembles here — likely untuned multiclass settings "
                       "on a heavily imbalanced set; worth a look before ruling it out.")
        else:
            st.info("model_comparison.csv not found.")
    with b:
        st.subheader("Top features (Random Forest importance)")
        if fi_df is not None:
            fi = fi_df.rename(columns={fi_df.columns[0]: "Feature", fi_df.columns[1]: "Importance"})
            n = st.slider("Features to show", 5, len(fi), 20)
            fig = px.bar(fi.head(n), x="Importance", y="Feature", orientation="h", color_discrete_sequence=[INFO])
            fig.update_layout(**PLOT, height=560, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("feature_importance.csv not found.")

    for img, title in (("shap_summary_plot.png", "SHAP summary"), ("shap_feature_importance.png", "SHAP importance"),
                       ("waterfall_plot.png", "Waterfall (single flow)")):
        p = data_dir / img
        if p.exists():
            st.image(str(p), caption=title)

# ================================================================== LIVE
with tabs[5]:
    st.subheader("Score new flows")
    st.caption("Upload a CSV/Parquet with the same feature columns used in training "
               "(without Label / ClassLabel / Target).")
    mp, cp = data_dir / "final_xgboost.pkl", data_dir / "confidence_calibrator.pkl"
    if not (mp.exists() and cp.exists()):
        st.info("final_xgboost.pkl and confidence_calibrator.pkl are needed in the data folder.")
    else:
        @st.cache_resource
        def load_models(m, c):
            return joblib.load(m), joblib.load(c)

        up = st.file_uploader("Flows", type=["csv", "parquet"], key="live")
        if up:
            try:
                new = pd.read_parquet(up) if up.name.endswith(".parquet") else pd.read_csv(up)
                model, calib = load_models(str(mp), str(cp))
                feats = list(getattr(model, "feature_names_in_", []))
                if feats:
                    miss = [f for f in feats if f not in new.columns]
                    if miss:
                        st.error(f"Missing {len(miss)} feature columns, e.g. {miss[:5]}")
                        st.stop()
                    X = new[feats].replace([np.inf, -np.inf], np.nan)
                else:
                    X = new.select_dtypes("number")
                proba = model.predict_proba(X)
                pred = proba.argmax(1)
                raw = np.clip(proba.max(1), 1e-6, 1 - 1e-6)
                cal = calib.predict_proba(np.log(raw / (1 - raw)).reshape(-1, 1))[:, 1]
                out = new.copy()
                out.insert(0, "Predicted", [cname(p) for p in pred])
                out.insert(1, "Calibrated_Confidence", cal)
                out.insert(2, "Action", np.where(cal >= threshold, "Automatic Alert", "Analyst Review"))
                m1, m2, m3 = st.columns(3)
                m1.metric("Flows", f"{len(out):,}")
                m2.metric("Auto alerts", f"{(out['Action'] == 'Automatic Alert').sum():,}")
                m3.metric("To analyst", f"{(out['Action'] == 'Analyst Review').sum():,}")
                st.dataframe(out.head(1000), use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download scored flows", out.to_csv(index=False).encode(),
                                   "scored_flows.csv", "text/csv")
            except Exception as e:  # noqa: BLE001
                st.error(f"Scoring failed: {e}")
