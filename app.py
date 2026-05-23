import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

# ============================================================================
# STREAMLIT APP: EHR Risk Prediction Pipeline Explorer
# ============================================================================

st.set_page_config(page_title="EHR Risk Prediction", layout="wide")

st.title("🏥 Clinical Risk Prediction Pipeline Explorer")
st.markdown("Interactive visualization of the diabetes high-risk event prediction system")

# ============================================================================
# SIDEBAR: Pipeline Configuration
# ============================================================================

st.sidebar.header("⚙️ Pipeline Configuration")

# Initialize or load pipeline
@st.cache_resource
def load_pipeline():
    """Load and cache the pipeline"""
    from ehr_pipeline import EHRPipeline

    BASE_DIR = Path(__file__).resolve().parent

    REPO_DIR = Path("G:/My Drive/MASAID/AI_Project_Group/repo")

    pipe = EHRPipeline(
        data_dir=REPO_DIR / "data" / "week_3" / "new_data" / "csv",
        model_path=REPO_DIR / "data" / "week_3" / "best_xgboost_model.pkl"
    )
    pipe.load_csvs().clean_notes().extract_notes().build_profiles()
    return pipe

with st.spinner("Loading pipeline..."):
    pipe = load_pipeline()

st.sidebar.success(f"✅ Loaded {len(pipe.profiles):,} patient profiles")

# ============================================================================
# TAB 1: Pipeline Overview
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Pipeline Overview",
    "👤 Patient Explorer",
    "🔍 Model Predictions",
    "📈 Model Performance",
    "🧠 Feature Importance",
    "📄 Course Rubric",
    "🎞️ Slide Deck"
])

with tab1:
    st.header("Pipeline Data Flow")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Patients", f"{len(pipe.profiles):,}")
    with col2:
        st.metric("Encounters", f"{len(pipe.encounters_df):,}" if pipe.encounters_df is not None else "N/A")
    with col3:
        st.metric("Observations", f"{len(pipe.observations_df):,}" if pipe.observations_df is not None else "N/A")
    with col4:
        st.metric("Medications", f"{len(pipe.medications_df):,}" if pipe.medications_df is not None else "N/A")
    
    st.markdown("---")
    
    # Pipeline stages
    st.subheader("1️⃣ Stage 1: Load CSVs")
    if st.checkbox("Show raw data sample"):
        data_source = st.selectbox("Select data source:", 
            ["Patients", "Encounters", "Conditions", "Observations", "Medications"])
        
        df_map = {
            "Patients": pipe.patients_df,
            "Encounters": pipe.encounters_df,
            "Conditions": pipe.conditions_df,
            "Observations": pipe.observations_df,
            "Medications": pipe.medications_df
        }
        
        df = df_map[data_source]
        if df is not None:
            st.dataframe(df.head(100), use_container_width=True)
            st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    st.subheader("2️⃣ Stage 2: Clean Notes")
    if pipe.notes_df is not None and st.checkbox("Show note cleaning"):
        st.write(f"Total notes: {len(pipe.notes_df):,}")
        sample_note = pipe.notes_df.sample(1).iloc[0]
        st.text_area("Sample cleaned note:", sample_note['note_text'][:500] + "...", height=150)
    
    st.subheader("3️⃣ Stage 3: Extract Structured Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Extracted Vitals", len(pipe.extracted_vitals))
    with col2:
        st.metric("Extracted Labs", len(pipe.extracted_labs))
    with col3:
        st.metric("Extracted Meds", len(pipe.extracted_meds))
    
    st.subheader("4️⃣ Stage 4: Build Patient Profiles")
    st.write(f"Created {len(pipe.profiles):,} ExtendedPatientProfile objects")

# ============================================================================
# TAB 2: Patient Explorer
# ============================================================================

with tab2:
    st.header("👤 Individual Patient Timeline Explorer")
    
    # Patient selector
    patient_ids = list(pipe.profiles.keys())
    selected_patient = st.selectbox("Select Patient:", patient_ids, index=0)
    
    if selected_patient:
        profile = pipe.profiles[selected_patient]
        
        # Patient summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("HbA1c Readings", len(profile.hba1c_timeline))
        with col2:
            st.metric("BP Readings", len(profile.bp_timeline))
        with col3:
            st.metric("Encounters", len(profile.encounter_timeline))
        with col4:
            st.metric("Medications", len(profile.medication_timeline))
        
        st.markdown("---")
        
        # Timeline visualization
        st.subheader("📈 Clinical Timelines")
        
        timeline_type = st.selectbox("Select timeline:", [
            "HbA1c Trend",
            "Blood Pressure",
            "eGFR",
            "Encounters",
            "Medications",
            "Emergency Visits",
            "Complications"
        ])
        
        if timeline_type == "HbA1c Trend" and len(profile.hba1c_timeline) > 0:
            hba1c_df = pd.DataFrame(profile.hba1c_timeline)
            fig = px.line(hba1c_df, x='date', y='value', 
                         title='HbA1c Over Time',
                         labels={'value': 'HbA1c (%)', 'date': 'Date'})
            fig.add_hline(y=7.0, line_dash="dash", line_color="orange", 
                         annotation_text="Target: 7.0%")
            fig.add_hline(y=9.0, line_dash="dash", line_color="red", 
                         annotation_text="High Risk: 9.0%")
            st.plotly_chart(fig, use_container_width=True)
        
        elif timeline_type == "Blood Pressure" and len(profile.bp_timeline) > 0:
            bp_df = pd.DataFrame(profile.bp_timeline)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bp_df['date'], y=bp_df['systolic'], 
                                    name='Systolic', mode='lines+markers'))
            fig.add_trace(go.Scatter(x=bp_df['date'], y=bp_df['diastolic'], 
                                    name='Diastolic', mode='lines+markers'))
            fig.add_hline(y=140, line_dash="dash", line_color="red", 
                         annotation_text="Systolic Target: 140")
            fig.update_layout(title='Blood Pressure Over Time',
                            xaxis_title='Date',
                            yaxis_title='mmHg')
            st.plotly_chart(fig, use_container_width=True)
        
        elif timeline_type == "Encounters" and len(profile.encounter_timeline) > 0:
            enc_df = pd.DataFrame(profile.encounter_timeline)
            enc_counts = enc_df.groupby(enc_df['date'].dt.to_period('M')).size()
            fig = px.bar(x=enc_counts.index.astype(str), y=enc_counts.values,
                        title='Encounters per Month',
                        labels={'x': 'Month', 'y': 'Number of Encounters'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature snapshot at a specific date
        st.markdown("---")
        st.subheader("📸 Feature Snapshot")
        
        # Date picker
        min_date = datetime(2010, 1, 1).date()
        max_date = datetime(2025, 12, 31).date()
        selected_date = st.date_input("Select date:", value=datetime(2015, 6, 15).date(),
                                     min_value=min_date, max_value=max_date)
        
        if st.button("Get Features"):
            try:
                features = pipe.get_patient_features(selected_patient, str(selected_date))
                
                # Display as table
                feature_df = pd.DataFrame([
                    {"Feature": k, "Value": v} for k, v in features.items()
                ])
                st.dataframe(feature_df, use_container_width=True, height=400)
                
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================================
# TAB 3: Model Predictions
# ============================================================================

with tab3:
    st.header("🔍 Risk Prediction & Explanation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        pred_patient = st.selectbox("Select Patient:", patient_ids, key="pred_patient")
    with col2:
        pred_date = st.date_input("Prediction Date:", value=datetime(2015, 6, 15).date(),
                                 key="pred_date")
    
    if st.button("🎯 Generate Prediction & Explanation", type="primary"):
        with st.spinner("Computing SHAP values..."):
            try:
                # Get prediction
                prob = pipe.predict_patient(pred_patient, str(pred_date))
                
                # Get explanation
                report = pipe.explain_patient(pred_patient, str(pred_date), top_n=5)
                
                # Display results
                st.markdown("---")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Risk Score", f"{report['risk_score']:.2%}")
                with col2:
                    risk_color = {
                        "LOW": "🟢",
                        "MODERATE": "🟡", 
                        "HIGH": "🟠",
                        "CRITICAL": "🔴"
                    }
                    st.metric("Risk Level", f"{risk_color[report['risk_level']]} {report['risk_level']}")
                with col3:
                    st.metric("Patient ID", pred_patient[:12] + "...")
                
                st.markdown("---")
                
                # Risk factors
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("⬆️ Risk Factors (Increase Risk)")
                    for i, factor in enumerate(report['risk_factors'], 1):
                        with st.expander(f"{i}. {factor['feature']} ({factor['contribution_pct']:.1f}%)"):
                            st.write(f"**Value:** {factor['value']}")
                            st.write(f"**Clinical Meaning:** {factor['clinical_meaning']}")
                            st.progress(factor['contribution_pct'] / 100)
                
                with col2:
                    st.subheader("⬇️ Protective Factors (Decrease Risk)")
                    for i, factor in enumerate(report['protective_factors'], 1):
                        with st.expander(f"{i}. {factor['feature']} ({factor['contribution_pct']:.1f}%)"):
                            st.write(f"**Value:** {factor['value']}")
                            st.write(f"**Clinical Meaning:** {factor['clinical_meaning']}")
                            st.progress(factor['contribution_pct'] / 100)
                
                # Waterfall chart
                st.markdown("---")
                st.subheader("📊 SHAP Waterfall (Top 10 Features)")
                
                all_factors = report['risk_factors'] + report['protective_factors']
                all_factors_sorted = sorted(all_factors, 
                                           key=lambda x: abs(x['contribution_pct']), 
                                           reverse=True)[:10]
                
                fig = go.Figure(go.Waterfall(
                    name="SHAP",
                    orientation="h",
                    y=[f['feature'] for f in all_factors_sorted],
                    x=[f['contribution_pct'] if f['direction'] == 'risk' else -f['contribution_pct'] 
                       for f in all_factors_sorted],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                ))
                fig.update_layout(title="Feature Contributions to Risk Score",
                                 xaxis_title="Contribution (%)",
                                 yaxis_title="Feature",
                                 height=500)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

# ============================================================================
# TAB 4: Model Performance
# ============================================================================

@st.cache_data
def get_feature_matrix(_pipe):
    return _pipe.build_feature_matrix(interval_days=30)


with tab4:
    st.header("📈 Model Performance Metrics")

    with st.spinner("Building feature matrix..."):
        feature_matrix = get_feature_matrix(pipe)
    
    st.success(f"✅ Feature matrix: {feature_matrix.shape[0]:,} rows × {feature_matrix.shape[1]} columns")
    
    # Evaluate model
    if st.button("🎯 Evaluate Model Performance"):
        with st.spinner("Computing metrics..."):
            metrics = pipe.evaluate_model(data=feature_matrix)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            with col2:
                st.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
            with col3:
                st.metric("Precision @ 80% Recall", f"{metrics['precision_at_80_recall']:.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Test Patients", f"{metrics['n_test_patients']:,}")
            with col2:
                st.metric("Test Samples", f"{metrics['n_test_rows']:,}")
            
            st.markdown("---")
            
            # Distribution of predictions
            st.subheader("📊 Prediction Distribution")
            
            from sklearn.model_selection import GroupShuffleSplit
            
            target_col = 'will_have_high_risk_event_next_30d'
            feature_cols = [c for c in feature_matrix.columns 
                           if c not in ['patient_id', 'date', target_col]]
            
            X = feature_matrix[feature_cols]
            y = feature_matrix[target_col]
            groups = feature_matrix['patient_id']
            
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(X, y, groups))
            
            y_prob = pipe.model.predict_proba(X.iloc[test_idx])[:, 1]
            y_test = y.iloc[test_idx]
            
            pred_df = pd.DataFrame({
                'probability': y_prob,
                'actual': y_test.map({0: 'Negative', 1: 'Positive'})
            })
            
            fig = px.histogram(pred_df, x='probability', color='actual',
                             nbins=50, barmode='overlay',
                             title='Distribution of Predicted Probabilities',
                             labels={'probability': 'Predicted Risk Probability'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Calibration plot
            st.subheader("📈 Calibration Curve")
            from sklearn.calibration import calibration_curve
            
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, 
                                    mode='markers+lines',
                                    name='Model'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                                    mode='lines',
                                    name='Perfect Calibration',
                                    line=dict(dash='dash', color='gray')))
            fig.update_layout(title='Calibration Curve',
                            xaxis_title='Predicted Probability',
                            yaxis_title='Actual Fraction of Positives')
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 5: Feature Importance
# ============================================================================

with tab5:
    st.header("🧠 Global Feature Importance")
    
    if st.button("📊 Compute SHAP Feature Importance"):
        with st.spinner("Computing SHAP values for test set..."):
            # Get test set
            from sklearn.model_selection import GroupShuffleSplit
            
            target_col = 'will_have_high_risk_event_next_30d'
            feature_cols = [c for c in feature_matrix.columns 
                           if c not in ['patient_id', 'date', target_col]]
            
            X = feature_matrix[feature_cols]
            y = feature_matrix[target_col]
            groups = feature_matrix['patient_id']
            
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(X, y, groups))
            
            X_test = X.iloc[test_idx]
            
            # Compute SHAP values (sample for speed)
            sample_size = min(1000, len(X_test))
            X_sample = X_test.sample(sample_size, random_state=42)
            X_sample = X_sample.astype('float32') 
            
            import shap
            explainer = shap.TreeExplainer(pipe.model)
            shap_values = explainer.shap_values(X_sample)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Global importance
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False).head(20)
            
            # Bar chart
            fig = px.bar(importance_df, x='importance', y='feature',
                        orientation='h',
                        title='Top 20 Most Important Features (Mean |SHAP|)',
                        labels={'importance': 'Mean |SHAP|', 'feature': 'Feature'})
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.dataframe(importance_df, use_container_width=True)

# ============================================================================
# TAB 6: Course Rubric PDF
# ============================================================================

with tab6:
    st.header("📄 AI Project Course — Grading Rubric")
    from streamlit_pdf_viewer import pdf_viewer
    pdf_viewer(str(Path(__file__).resolve().parent / "AI Project Course - Grading Rubric.pdf"))

with tab7:
    st.header("🎞️ Project Slide Deck")
    st.components.v1.iframe(
        "https://docs.google.com/presentation/d/e/2PACX-1vTDU-qlLMCRi6n7AFxwuPKPquaKPmDYnfpf7Mnj6XuaNrq6JyrabAMCN6yJqIYyqsyTxdPOEYUwCUM5/pubembed?start=true&loop=false&delayms=3000",
        height=600,
        scrolling=False,
    )

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption("🏥 Clinical Risk Prediction Pipeline | Built with Streamlit")