import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Telecom Customer Churn Intelligence Dashboard")
st.caption("Customer churn analytics • Risk prioritisation • Retention recommendations")

DATA_FILE = "dashboard_data.csv"
METRICS_FILE = "final_test_metrics.csv"
MODEL_FILE = "model_comparison.csv"
IMPORTANCE_FILE = "feature_importance.csv"

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

if not os.path.exists(DATA_FILE):
    st.error(
        f"Missing {DATA_FILE}. Run the dashboard-export cell in your Jupyter notebook first, "
        "then place the generated CSV files in the same folder as this app."
    )
    st.stop()

df = load_csv(DATA_FILE)

required = ["Actual_Churn", "Predicted_Churn", "Churn_Probability", "Risk_Level"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"dashboard_data.csv is missing required columns: {missing}")
    st.stop()

# Normalize common display fields
df["Churn_Probability"] = pd.to_numeric(df["Churn_Probability"], errors="coerce")
df["Risk_Level"] = df["Risk_Level"].astype(str)
df["Actual_Churn"] = df["Actual_Churn"].astype(str)
df["Predicted_Churn"] = df["Predicted_Churn"].astype(str)

with st.sidebar:
    st.header("Filters")
    view = df.copy()
    if "Contract" in view.columns:
        contracts = sorted(view["Contract"].dropna().astype(str).unique().tolist())
        selected_contracts = st.multiselect("Contract type", contracts, default=contracts)
        view = view[view["Contract"].astype(str).isin(selected_contracts)]
    risk_options = sorted(view["Risk_Level"].dropna().unique().tolist())
    selected_risks = st.multiselect("Risk level", risk_options, default=risk_options)
    view = view[view["Risk_Level"].isin(selected_risks)]
    if "InternetService" in view.columns:
        services = sorted(view["InternetService"].dropna().astype(str).unique().tolist())
        selected_services = st.multiselect("Internet service", services, default=services)
        view = view[view["InternetService"].astype(str).isin(selected_services)]

total = len(view)
actual_churn_rate = (view["Actual_Churn"].str.lower() == "yes").mean() if total else 0
predicted_churn = (view["Predicted_Churn"].str.lower() == "yes").sum() if total else 0
high_risk_count = (view["Risk_Level"].str.lower() == "high").sum() if total else 0
mean_probability = view["Churn_Probability"].mean() if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers in view", f"{total:,}")
c2.metric("Actual churn rate", f"{actual_churn_rate:.1%}")
c3.metric("Predicted churners", f"{predicted_churn:,}")
c4.metric("High-risk customers", f"{high_risk_count:,}")

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Actual vs predicted churn")
    counts = pd.DataFrame({
        "Status": ["Not Churned", "Churned"],
        "Actual": [
            (view["Actual_Churn"].str.lower() == "no").sum(),
            (view["Actual_Churn"].str.lower() == "yes").sum()
        ],
        "Predicted": [
            (view["Predicted_Churn"].str.lower() == "no").sum(),
            (view["Predicted_Churn"].str.lower() == "yes").sum()
        ]
    })
    melted = counts.melt(id_vars="Status", var_name="Type", value_name="Customers")
    fig = px.bar(melted, x="Status", y="Customers", color="Type", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Customer risk distribution")
    risk_counts = view["Risk_Level"].value_counts().rename_axis("Risk level").reset_index(name="Customers")
    if not risk_counts.empty:
        fig = px.pie(risk_counts, names="Risk level", values="Customers", hole=0.45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No customers match the selected filters.")

st.subheader("Churn probability distribution")
if not view.empty:
    fig = px.histogram(
        view, x="Churn_Probability", color="Actual_Churn",
        nbins=20, barmode="overlay", opacity=0.7,
        labels={"Churn_Probability": "Predicted churn probability", "Actual_Churn": "Actual churn"}
    )
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Churn by contract")
    if "Contract" in view.columns and not view.empty:
        contract_summary = view.groupby("Contract", dropna=False).agg(
            Customers=("Actual_Churn", "size"),
            Actual_Churn_Rate=("Actual_Churn", lambda s: (s.astype(str).str.lower() == "yes").mean()),
            Average_Churn_Probability=("Churn_Probability", "mean")
        ).reset_index()
        fig = px.bar(contract_summary, x="Contract", y="Actual_Churn_Rate",
                     hover_data=["Customers", "Average_Churn_Probability"],
                     labels={"Actual_Churn_Rate": "Actual churn rate"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Contract column not available.")

with right:
    st.subheader("Feature importance")
    if os.path.exists(IMPORTANCE_FILE):
        imp = load_csv(IMPORTANCE_FILE)
        if {"Feature", "Importance"}.issubset(imp.columns):
            imp = imp.sort_values("Importance", ascending=False).head(12).sort_values("Importance")
            fig = px.bar(imp, x="Importance", y="Feature", orientation="h")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance file does not have Feature and Importance columns.")
    else:
        st.info("Add feature_importance.csv to show the feature importance chart.")

st.subheader("High-risk customer action list")
risk_view = view[
    (view["Risk_Level"].str.lower() == "high") &
    (view["Predicted_Churn"].str.lower() == "yes")
].copy().sort_values("Churn_Probability", ascending=False)

display_cols = [
    c for c in [
        "customerID", "CustomerID", "tenure", "Contract", "MonthlyCharges",
        "Churn_Probability", "Risk_Level", "Retention_Recommendation", "Actual_Churn"
    ] if c in risk_view.columns
]
if not display_cols:
    display_cols = list(risk_view.columns)
st.dataframe(risk_view[display_cols].head(200), use_container_width=True, hide_index=True)

st.download_button(
    "Download filtered customer results (CSV)",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name="filtered_churn_results.csv",
    mime="text/csv"
)

if os.path.exists(METRICS_FILE):
    st.subheader("Final test-set model metrics")
    metrics = load_csv(METRICS_FILE)
    st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)

if os.path.exists(MODEL_FILE):
    st.subheader("Cross-validation model comparison")
    model_results = load_csv(MODEL_FILE)
    st.dataframe(model_results.round(4), use_container_width=True, hide_index=True)

st.caption(
    "Interpretation note: risk thresholds are illustrative. Model outputs indicate predicted risk, "
    "not certainty or proof that a customer will churn. Validate thresholds before operational use."
)

# ================= CUSTOMER ANALYTICS =================

import plotly.express as px
import pandas as pd
import streamlit as st

st.divider()
st.header("Customer Analytics")
st.caption(
    "Explore churn patterns across contracts, tenure, services, "
    "monthly charges and payment methods."
)

# Use the filtered customer view already created by the dashboard
analytics_df = view.copy()

# Convert churn labels into a consistent format
if "Actual_Churn" in analytics_df.columns:
    churn_col = "Actual_Churn"
elif "Churn" in analytics_df.columns:
    churn_col = "Churn"
else:
    churn_col = None

if churn_col is None:
    st.warning("Actual churn column is missing from the dashboard data.")
else:
    analytics_df[churn_col] = (
        analytics_df[churn_col].astype(str).str.strip()
    )

    # 1. Churn by contract
    if "Contract" in analytics_df.columns:
        st.subheader("1. Churn by Contract")

        contract_summary = (
            analytics_df.groupby("Contract", dropna=False)
            .agg(
                Customers=(churn_col, "size"),
                Churned=(churn_col, lambda s: s.eq("Yes").sum())
            )
            .reset_index()
        )

        contract_summary["Churn Rate (%)"] = (
            100 * contract_summary["Churned"]
            / contract_summary["Customers"].clip(lower=1)
        ).round(1)

        fig = px.bar(
            contract_summary,
            x="Contract",
            y="Churn Rate (%)",
            text="Churn Rate (%)",
            hover_data=["Customers", "Churned"],
            title="Actual churn rate by contract"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(contract_summary, hide_index=True)

    # 2. Churn by tenure group
    if "TenureGroup" in analytics_df.columns:
        st.subheader("2. Churn by Tenure Group")

        tenure_summary = (
            analytics_df.groupby("TenureGroup", dropna=False)
            .agg(
                Customers=(churn_col, "size"),
                Churned=(churn_col, lambda s: s.eq("Yes").sum())
            )
            .reset_index()
        )

        tenure_summary["Churn Rate (%)"] = (
            100 * tenure_summary["Churned"]
            / tenure_summary["Customers"].clip(lower=1)
        ).round(1)

        fig = px.bar(
            tenure_summary,
            x="TenureGroup",
            y="Churn Rate (%)",
            text="Churn Rate (%)",
            title="Actual churn rate by tenure group"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tenure_summary, hide_index=True)

    # 3. Churn by internet service
    if "InternetService" in analytics_df.columns:
        st.subheader("3. Churn by Internet Service")

        service_summary = (
            analytics_df.groupby("InternetService", dropna=False)
            .agg(
                Customers=(churn_col, "size"),
                Churned=(churn_col, lambda s: s.eq("Yes").sum())
            )
            .reset_index()
        )

        service_summary["Churn Rate (%)"] = (
            100 * service_summary["Churned"]
            / service_summary["Customers"].clip(lower=1)
        ).round(1)

        fig = px.bar(
            service_summary,
            x="InternetService",
            y="Churn Rate (%)",
            color="InternetService",
            text="Churn Rate (%)",
            title="Actual churn rate by internet service"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 4. Monthly charges by actual churn outcome
    if "MonthlyCharges" in analytics_df.columns:
        st.subheader("4. Monthly Charges vs Churn")

        charge_df = analytics_df.copy()
        charge_df["MonthlyCharges"] = pd.to_numeric(
            charge_df["MonthlyCharges"], errors="coerce"
        )
        charge_df = charge_df.dropna(
            subset=["MonthlyCharges", churn_col]
        )

        fig = px.box(
            charge_df,
            x=churn_col,
            y="MonthlyCharges",
            color=churn_col,
            points="outliers",
            title="Monthly charge distribution by actual churn"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 5. Churn by payment method
    if "PaymentMethod" in analytics_df.columns:
        st.subheader("5. Churn by Payment Method")

        payment_summary = (
            analytics_df.groupby("PaymentMethod", dropna=False)
            .agg(
                Customers=(churn_col, "size"),
                Churned=(churn_col, lambda s: s.eq("Yes").sum())
            )
            .reset_index()
        )

        payment_summary["Churn Rate (%)"] = (
            100 * payment_summary["Churned"]
            / payment_summary["Customers"].clip(lower=1)
        ).round(1)

        fig = px.bar(
            payment_summary,
            x="PaymentMethod",
            y="Churn Rate (%)",
            text="Churn Rate (%)",
            title="Actual churn rate by payment method"
        )
        st.plotly_chart(fig, use_container_width=True)

st.caption(
    "These charts describe observed customer churn patterns; "
    "they do not establish that a feature causes churn."
)

# ================= CUSTOMER RISK EXPLORER =================

st.divider()
st.header("🔎 Individual Customer Risk Explorer")
st.write(
    "Select a customer to view their saved churn prediction "
    "and a suggested retention action."
)

# Use the dashboard's current filtered dataset
risk_df = view.copy()

# Check required columns before displaying the explorer
required_cols = [
    "Predicted_Churn",
    "Risk_Level",
    "Churn_Probability"
]

missing_cols = [c for c in required_cols if c not in risk_df.columns]

if missing_cols:
    st.warning(
        "Some prediction columns are missing: "
        + ", ".join(missing_cols)
        + ". Check that app.py loads the updated dashboard_data.csv."
    )
else:
    # Find customer identifier column
    if "customerID" in risk_df.columns:
        id_col = "customerID"
    elif "CustomerID" in risk_df.columns:
        id_col = "CustomerID"
    else:
        id_col = None

    if risk_df.empty:
        st.info("No customers match the selected filters.")
    elif id_col is None:
        st.warning("Customer ID column was not found.")
    else:
        customer_ids = (
            risk_df[id_col].dropna().astype(str).unique().tolist()
        )

        selected_id = st.selectbox(
            "Select Customer ID",
            customer_ids,
            key="risk_explorer_customer"
        )

        customer = risk_df[
            risk_df[id_col].astype(str) == selected_id
        ].iloc[0]

        probability = pd.to_numeric(
            pd.Series([customer["Churn_Probability"]]),
            errors="coerce"
        ).iloc[0]

        predicted = str(customer["Predicted_Churn"])
        risk_level = str(customer["Risk_Level"])

        col1, col2, col3 = st.columns(3)

        col1.metric("Predicted churn", predicted)

        if pd.notna(probability):
            # Support probabilities saved as either 0–1 or 0–100
            display_probability = (
                probability * 100 if probability <= 1 else probability
            )
            col2.metric(
                "Churn probability",
                f"{display_probability:.1f}%"
            )
        else:
            col2.metric("Churn probability", "Unavailable")

        col3.metric("Risk level", risk_level)

        st.subheader("Customer Details")

        detail_cols = [
            c for c in [
                "Contract",
                "tenure",
                "TenureGroup",
                "InternetService",
                "MonthlyCharges",
                "PaymentMethod",
                "Actual_Churn"
            ]
            if c in risk_df.columns
        ]

        if detail_cols:
            st.dataframe(
                customer[detail_cols].to_frame("Value"),
                use_container_width=True
            )

        st.subheader("Suggested Retention Action")

        if "Retention_Recommendation" in risk_df.columns:
            recommendation = customer["Retention_Recommendation"]
            st.info(str(recommendation))
        elif risk_level.lower() == "high":
            st.info(
                "Prioritise a customer-service follow-up and review "
                "suitable contract or service options."
            )
        elif risk_level.lower() == "medium":
            st.info(
                "Review customer feedback and consider a personalised "
                "service check-in."
            )
        else:
            st.info(
                "Continue regular customer engagement and monitor "
                "future changes in risk."
            )

        st.caption(
            "Predictions are estimates from the trained model, "
            "not guarantees that a customer will leave."
        )

# ================= MODEL EVALUATION DASHBOARD =================

st.divider()
st.header("🧠 Model Evaluation & Result Analysis")
st.caption(
    "Compare classification algorithms using a separate test set."
)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import numpy as np
import matplotlib.pyplot as plt

# Use the full dataset where available
if "df" in globals():
    eval_df = df.copy()
else:
    eval_df = view.copy()

# Identify the actual target column
if "Churn" in eval_df.columns:
    target_col = "Churn"
elif "Actual_Churn" in eval_df.columns:
    target_col = "Actual_Churn"
else:
    target_col = None

if target_col is None:
    st.warning(
        "Actual churn target column was not found. "
        "Model evaluation requires the real Churn labels."
    )
else:
    eval_df[target_col] = (
        eval_df[target_col].astype(str).str.strip()
    )
    eval_df = eval_df[
        eval_df[target_col].isin(["Yes", "No", "1", "0"])
    ].copy()

    # Exclude identifiers, target and previously generated predictions
    exclude_cols = [
        target_col, "customerID", "CustomerID",
        "Churn_Flag", "Predicted_Churn",
        "Churn_Probability", "Risk_Level",
        "Actual_Churn", "Retention_Recommendation"
    ]

    # Use original customer features only
    X_eval = eval_df.drop(
        columns=[c for c in exclude_cols if c in eval_df.columns],
        errors="ignore"
    )
    y_eval = eval_df[target_col].map(
        {"No": 0, "Yes": 1, "0": 0, "1": 1}
    )

    # Remove rows with invalid target values
    valid_rows = y_eval.notna()
    X_eval = X_eval.loc[valid_rows].copy()
    y_eval = y_eval.loc[valid_rows].astype(int)

    if y_eval.nunique() < 2 or len(y_eval) < 20:
        st.warning(
            "Not enough valid records or target classes "
            "to evaluate the models."
        )
    else:
        # Convert numeric-looking columns safely
        for col in X_eval.columns:
            if X_eval[col].dtype == "object":
                converted = pd.to_numeric(
                    X_eval[col], errors="coerce"
                )
                if converted.notna().sum() > 0 and (
                    converted.notna().sum()
                    >= X_eval[col].notna().sum() * 0.9
                ):
                    X_eval[col] = converted

        numeric_cols = X_eval.select_dtypes(
            include=["number"]
        ).columns.tolist()
        categorical_cols = [
            c for c in X_eval.columns if c not in numeric_cols
        ]

        transformers = []

        if numeric_cols:
            numeric_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            transformers.append(
                ("num", numeric_pipe, numeric_cols)
            )

        if categorical_cols:
            categorical_pipe = Pipeline([
                ("imputer", SimpleImputer(
                    strategy="most_frequent"
                )),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore"
                ))
            ])
            transformers.append(
                ("cat", categorical_pipe, categorical_cols)
            )

        preprocessor_eval = ColumnTransformer(transformers)

        X_train_eval, X_test_eval, y_train_eval, y_test_eval = (
            train_test_split(
                X_eval, y_eval,
                test_size=0.2,
                random_state=42,
                stratify=y_eval
            )
        )

        models_eval = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=8,
                class_weight="balanced",
                random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        }

        trained_eval_models = {}
        predictions_eval = {}
        probabilities_eval = {}
        metric_rows = []

        with st.spinner("Training and evaluating models..."):
            for model_name, algorithm in models_eval.items():
                model_pipe = Pipeline([
                    ("preprocessor", preprocessor_eval),
                    ("model", algorithm)
                ])

                model_pipe.fit(X_train_eval, y_train_eval)

                pred = model_pipe.predict(X_test_eval)
                prob = model_pipe.predict_proba(X_test_eval)[:, 1]

                trained_eval_models[model_name] = model_pipe
                predictions_eval[model_name] = pred
                probabilities_eval[model_name] = prob

                metric_rows.append({
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_test_eval, pred),
                    "Precision": precision_score(
                        y_test_eval, pred, zero_division=0
                    ),
                    "Recall": recall_score(
                        y_test_eval, pred, zero_division=0
                    ),
                    "F1-score": f1_score(
                        y_test_eval, pred, zero_division=0
                    ),
                    "ROC-AUC": roc_auc_score(
                        y_test_eval, prob
                    )
                })

        metrics_df = pd.DataFrame(metric_rows)
        metrics_df = metrics_df.sort_values(
            "F1-score", ascending=False
        ).reset_index(drop=True)

        st.success(
            f"Evaluation completed on {len(y_test_eval):,} "
            "held-out test records."
        )

        st.subheader("1. Algorithm Performance Comparison")
        st.dataframe(
            metrics_df.style.format({
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1-score": "{:.3f}",
                "ROC-AUC": "{:.3f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        metric_choice = st.selectbox(
            "Choose metric to compare",
            ["F1-score", "ROC-AUC", "Recall",
             "Precision", "Accuracy"],
            key="evaluation_metric_choice"
        )

        metric_chart = metrics_df.melt(
            id_vars="Model",
            value_vars=[
                "Accuracy", "Precision", "Recall",
                "F1-score", "ROC-AUC"
            ],
            var_name="Metric",
            value_name="Score"
        )

        selected_chart = metric_chart[
            metric_chart["Metric"] == metric_choice
        ]

        fig_compare = px.bar(
            selected_chart,
            x="Model",
            y="Score",
            text=selected_chart["Score"].round(3),
            title=f"Model comparison — {metric_choice}",
            range_y=[0, 1]
        )
        st.plotly_chart(
            fig_compare, use_container_width=True
        )

        best_name = metrics_df.iloc[0]["Model"]
        best_model = trained_eval_models[best_name]
        best_pred = predictions_eval[best_name]

        st.subheader("2. Best Model on F1-score")
        st.write(f"**Selected model:** {best_name}")
        st.write(
            "F1-score balances precision and recall, which is useful "
            "when identifying customers who may churn."
        )

        st.subheader("3. Confusion Matrix")
        cm = confusion_matrix(
            y_test_eval, best_pred, labels=[0, 1]
        )

        fig_cm, ax_cm = plt.subplots()
        ax_cm.imshow(cm)
        ax_cm.set_title(f"Confusion Matrix — {best_name}")
        ax_cm.set_xlabel("Predicted label")
        ax_cm.set_ylabel("Actual label")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Not Churned", "Churned"])
        ax_cm.set_yticklabels(["Not Churned", "Churned"])

        for i in range(2):
            for j in range(2):
                ax_cm.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center"
                )

        st.pyplot(fig_cm)
        plt.close(fig_cm)

        st.subheader("4. Classification Report")
        report = classification_report(
            y_test_eval,
            best_pred,
            labels=[0, 1],
            target_names=["Not Churned", "Churned"],
            output_dict=True,
            zero_division=0
        )
        st.dataframe(
            pd.DataFrame(report).transpose().round(3),
            use_container_width=True
        )

        st.subheader("5. Important Churn Features")
        fitted_preprocessor = best_model.named_steps[
            "preprocessor"
        ]
        fitted_algorithm = best_model.named_steps["model"]

        feature_names = fitted_preprocessor.get_feature_names_out()

        if hasattr(fitted_algorithm, "feature_importances_"):
            importance_values = fitted_algorithm.feature_importances_
        elif hasattr(fitted_algorithm, "coef_"):
            importance_values = np.abs(
                fitted_algorithm.coef_[0]
            )
        else:
            importance_values = None

        if importance_values is not None:
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance_values
            }).sort_values(
                "Importance", ascending=False
            ).head(15)

            fig_importance = px.bar(
                importance_df.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 influential features"
            )
            st.plotly_chart(
                fig_importance, use_container_width=True
            )
        else:
            st.info("Feature importance is unavailable for this model.")

        st.subheader("6. Download Evaluation Results")
        st.download_button(
            "Download model comparison CSV",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="telecom_model_evaluation.csv",
            mime="text/csv"
        )

        st.caption(
            "Metrics are calculated from a held-out test set. "
            "Feature importance indicates model association, "
            "not proof of causation."
        )
