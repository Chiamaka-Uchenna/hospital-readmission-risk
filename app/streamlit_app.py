"""Hospital readmission risk: interactive app.

Run with:  streamlit run app/streamlit_app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

from common import BLUE, ORANGE, bar, lines, load, predict, show, vector

st.set_page_config(page_title="Hospital readmission risk", layout="wide")
model, s = load("model.json"), load("summary.json")
base = s["base_rate"]
gbm = next(m for m in s["metrics"] if m["model"] == "Gradient boosting")

LABELS = {
    "age": "Age", "time_in_hospital": "Days in hospital", "num_lab_procedures": "Lab procedures", "num_procedures": "Other procedures",
    "num_medications": "Distinct medications", "number_diagnoses": "Diagnoses recorded", "prior_inpatient": "Inpatient stays, past year",
    "prior_emergency": "Emergency visits, past year", "prior_outpatient": "Outpatient visits, past year",
    "diabetes_drugs": "Diabetes drugs prescribed", "dose_changes": "Diabetes drug dose changes", "admission_type": "Admission type",
    "admission_source": "Admitted from", "discharge": "Discharged to", "primary_diagnosis": "Primary diagnosis", "a1c": "HbA1c result",
    "insulin": "Insulin",
}

st.title("Who comes back to hospital within 30 days?")
st.caption("A readmission risk model for patients with diabetes, trained on real encounters from 130 US hospitals. "
           "Research demonstration only. Not a clinical tool.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients analysed", f"{s['patients']:,}")
c2.metric("Readmitted within 30 days", f"{base:.1%}")
c3.metric("Model AUROC", f"{gbm['AUROC']:.3f}")
c4.metric("Top risk decile", f"{s['lift'][0]['lift']:.1f}x average")

tab_score, tab_explore, tab_model, tab_plan = st.tabs(["Risk calculator", "Who gets readmitted", "Model performance", "Targeting follow-up"])

with tab_score:
    st.subheader("Describe a patient at discharge")
    values = {}
    columns = st.columns(3)
    position = 0
    for field in s["schema"]["numeric"]:
        column = columns[position % 3]
        position += 1
        name = field["name"]
        if name == "female":
            values[name] = 1 if column.selectbox("Sex", ["Female", "Male"]) == "Female" else 0
        elif name == "on_diabetes_medication":
            values[name] = 1 if column.selectbox("On diabetes medication", ["Yes", "No"]) == "Yes" else 0
        else:
            values[name] = column.number_input(LABELS.get(name, name), min_value=field["min"], max_value=field["max"], value=field["default"],
                                               step=10 if name == "age" else 1)
    for field in s["schema"]["categorical"]:
        column = columns[position % 3]
        position += 1
        choice = column.selectbox(LABELS.get(field["name"], field["name"]), field["levels"], index=field["levels"].index(field["default"]))
        values[f"{field['name']}__{choice.replace(' ', '_')}"] = 1

    risk = predict(model, vector(model, values))
    quantiles = np.array(s["risk_quantiles"])
    percentile = int(np.searchsorted(quantiles, risk))
    band = "High risk" if risk >= quantiles[90] else "Elevated risk" if risk >= quantiles[60] else "Lower risk"
    st.divider()
    left, right = st.columns([1, 2])
    left.metric("Estimated 30-day readmission risk", f"{risk:.1%}", band, delta_color="off")
    right.progress(min(1.0, risk / quantiles[100]))
    right.write(f"{risk / base:.1f} times the average risk of {base:.1%}. Higher than about {min(percentile, 100)} percent of patients in the test set.")

with tab_explore:
    st.caption("Share of patients readmitted within 30 days, by group. The dotted line is the overall rate.")
    left, right = st.columns(2)
    for column, key, field, title in [(left, "rate_by_age", "age", "Age (bracket midpoint)"),
                                      (right, "rate_by_prior_inpatient", "prior_inpatient", "Inpatient stays in the previous year")]:
        rows = s[key]
        with column:
            show(bar([str(r[field]) for r in rows], [r["rate"] * 100 for r in rows], title, "Readmitted (%)", reference=base * 100))
    left, right = st.columns(2)
    for column, key, field in [(left, "rate_by_discharge", "discharge"), (right, "rate_by_diagnosis", "primary_diagnosis")]:
        rows = s[key]
        with column:
            show(bar([r[field] for r in rows], [r["rate"] * 100 for r in rows], "Readmitted (%)", "", horizontal=True, reference=base * 100))

with tab_model:
    st.caption(f"Evaluated on {s['test_patients']:,} held-out patients.")
    left, right = st.columns(2)
    cal = s["calibration"]
    with left:
        st.markdown("**Predicted against observed risk**")
        top = max(cal["predicted"] + cal["observed"]) * 108
        show(lines([("Model", [v * 100 for v in cal["predicted"]], [v * 100 for v in cal["observed"]], BLUE, "solid")],
                   "Predicted risk (%)", "Observed rate (%)", diagonal=top))
    with right:
        st.markdown("**Observed readmission rate by risk decile**")
        show(bar([str(r["decile"]) for r in s["lift"]], [r["rate"] * 100 for r in s["lift"]], "Risk decile (1 = highest)", "Readmitted (%)", reference=base * 100))
    left, right = st.columns(2)
    with left:
        st.markdown("**What drives the predictions**")
        importance = s["importance"][::-1]
        show(bar([r["feature"] for r in importance], [r["value"] for r in importance], "Mean absolute SHAP value", "", horizontal=True))
    with right:
        st.markdown("**Performance across patient groups**")
        fairness = pd.DataFrame(s["fairness"])[["group", "patients", "observed", "predicted", "auroc"]]
        st.dataframe(fairness.style.format({"observed": "{:.1%}", "predicted": "{:.1%}", "auroc": "{:.3f}", "patients": "{:,}"}), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(s["metrics"]).round(4), hide_index=True, width="stretch")

with tab_plan:
    st.caption("A follow-up programme is offered to discharged patients, highest risk first. Change the assumptions to see how many patients it is worth enrolling.")
    a, b, c = st.columns(3)
    cost = a.slider("Programme cost per patient ($)", 50, 1500, 300, 50)
    readmission_cost = b.slider("Cost of one readmission ($)", 5000, 30000, 15000, 1000)
    effect = c.slider("Readmissions prevented among enrolled (%)", 5, 50, 20, 5) / 100
    outcomes = np.array(s["ranked_outcomes"])
    enrolled = np.arange(1, len(outcomes) + 1)
    net = outcomes.cumsum() * effect * readmission_cost - enrolled * cost
    best = int(net.argmax())
    figure = lines([("Net saving", enrolled / len(outcomes) * 100, net / 1000, BLUE, "solid")],
                   "Share of patients enrolled, highest risk first (%)", "Net saving (thousand $)")
    figure.add_scatter(x=[enrolled[best] / len(outcomes) * 100], y=[net[best] / 1000], mode="markers", name="Best policy", marker=dict(color=ORANGE, size=11))
    show(figure)
    if net[best] <= 0:
        st.write("With these assumptions the programme does not pay for itself at any enrolment level.")
    else:
        st.write(f"Best policy: enrol the highest risk {enrolled[best] / len(outcomes):.0%} of patients, for a net saving of "
                 f"${net[best]:,.0f} across {len(outcomes):,} test patients. Enrolling everyone gives ${net[-1]:,.0f}.")
