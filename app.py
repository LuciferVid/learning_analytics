import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# path setup for local imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import load_and_preprocess_data
from recommendations import generate_recommendations
from study_coach import CoachInputs, generate_coach_report
from pdf_export import report_markdown_to_pdf_bytes

st.set_page_config(page_title="Learning Analytics Dashboard", layout="wide")

# some custom styling to make it look nice
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background-color: #4A90E2;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_model_stuff():
    try:
        with open('models/model.pkl', 'rb') as f:
            m = pickle.load(f)
        with open('models/target_mapping.pkl', 'rb') as f:
            tm = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            sc = pickle.load(f)
        with open('models/encoders.pkl', 'rb') as f:
            enc = pickle.load(f)
        return m, tm, sc, enc
    except:
        return None, None, None, None

model, mapping, scaler, encoders = get_model_stuff()
rev_map = {v: k for k, v in mapping.items()} if mapping else {}

# Simple sidebar navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Batch Data", "Get Recommendations", "AI Study Coach", "Info"])

if menu == "Home":
    st.title("Student Performance Analytics")
    st.write("Dashboard for analyzing student scores and predicting risks.")
    
    c1, c2, c3 = st.columns(3)
    c1.info("**Status:** Running")
    c2.success("**Accuracy:** ~70%")
    c3.warning("**Model:** LogReg")

    st.divider()
    st.image("https://images.unsplash.com/photo-1434030216411-0b793f4b4173?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80")

elif menu == "Batch Data":
    st.header("Analyze Multiple Students")
    f = st.file_uploader("Drop your CSV here", type="csv")
    
    if f:
        f.seek(0)
        data = pd.read_csv(f)
        t1, t2 = st.tabs(["Raw Data", "Plots"])
        
        with t1:
            st.write("Showing first 10 rows:")
            st.dataframe(data.head(10))
            
        with t2:
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("Distribution of Scores")
                fig1, ax1 = plt.subplots()
                sns.boxplot(data=data[['math score', 'reading score', 'writing score']], ax=ax1)
                st.pyplot(fig1)
            with col_b:
                st.write("Parents Education")
                fig2, ax2 = plt.subplots()
                data['parental level of education'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax2)
                st.pyplot(fig2)
            
            st.write("---")
            st.write("### Category Analysis")
            fig3, ax3 = plt.subplots()
            # Derive the category if not present
            if 'performance_category' not in data.columns:
                def get_grade(score):
                    if score >= 80: return 'High-performing'
                    if score >= 50: return 'Average'
                    return 'At-risk'
                data['total_score'] = data['math score'] + data['reading score'] + data['writing score']
                data['performance_category'] = (data['total_score'] / 3).apply(get_grade)
            
            data['performance_category'].value_counts().plot.bar(ax=ax3, color=['#4CAF50','#FFC107','#F44336'])
            plt.xticks(rotation=0)
            st.pyplot(fig3)
            
            st.write("---")
            st.write("### Detailed Score Distributions")
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=data[['math score', 'reading score', 'writing score']], ax=ax4, palette='Set3')
            ax4.set_title("Box Plot of Study Scores")
            st.pyplot(fig4)
        
        if st.button("Predict Risks"):
            if model:
                f.seek(0)
                proc_data, _ = load_and_preprocess_data(f)
                feat = proc_data.drop(['performance_category', 'total_score', 'percentage', 'math score', 'reading score', 'writing score'], axis=1)
                scaled_feat = scaler.transform(feat)
                
                out = model.predict(scaled_feat)
                data['Risk Level'] = [rev_map[o] for o in out]
                
                st.success("Done!")
                st.dataframe(data[['gender', 'race/ethnicity', 'Risk Level']])
                
                # add a graph for predicted risks
                st.write("### Predicted Risk Breakdown")
                fig_out, ax_out = plt.subplots()
                data['Risk Level'].value_counts().plot.bar(ax=ax_out, color=['#ff9999','#66b3ff','#99ff99'])
                plt.xticks(rotation=45)
                st.pyplot(fig_out)
            else:
                st.error("Model files missing in models/ folder")

elif menu == "Get Recommendations":
    st.header("Individual Student Coach")
    st.write("Enter scores to get a custom study plan.")
    
    with st.form("my_form"):
        m_score = st.slider("Math", 0, 100, 50)
        r_score = st.slider("Reading", 0, 100, 50)
        w_score = st.slider("Writing", 0, 100, 50)
        submit = st.form_submit_button("Analyze")
        
    if submit:
        final_avg = (m_score + r_score + w_score) / 3
        st.write(f"Average Score: {final_avg:.2f}%")
        
        tips = generate_recommendations({
            'math score': m_score,
            'reading score': r_score,
            'writing score': w_score
        })
        
        for t in tips:
            st.info(t)

elif menu == "AI Study Coach":
    st.header("Agentic AI Study Coach (Milestone 2)")
    st.write("Diagnose learning gaps, build a multi-week plan, and retrieve learning resources (free).")

    left, right = st.columns([1, 1])
    with left:
        goal = st.text_input("Student goal", value="Prepare for upcoming exams and improve weak areas")
        preferred_style = st.selectbox(
            "Preferred study style",
            ["Balanced (review + practice)", "Practice-heavy", "Concept-heavy", "Short daily sessions"],
            index=0,
        )
        hours_per_week = st.slider("Available study time (hours/week)", 1, 30, 6)
        weeks = st.slider("Time horizon (weeks)", 1, 12, 4)

    with right:
        st.subheader("Current scores")
        m_score = st.slider("Math score", 0, 100, 50, key="m2_math")
        r_score = st.slider("Reading score", 0, 100, 50, key="m2_reading")
        w_score = st.slider("Writing score", 0, 100, 50, key="m2_writing")

    c1, c2 = st.columns([1, 1])
    with c1:
        run = st.button("Generate Coach Report", type="primary")
    with c2:
        st.caption("Tip: set `HF_API_TOKEN` for faster hosted inference; otherwise it will try a local open-source model.")

    if run:
        inputs = CoachInputs(
            goal=goal.strip() or "Improve learning outcomes",
            hours_per_week=int(hours_per_week),
            weeks=int(weeks),
            math_score=int(m_score),
            reading_score=int(r_score),
            writing_score=int(w_score),
            preferred_style=preferred_style,
        )

        with st.spinner("Thinking, planning, and retrieving resources..."):
            result = generate_coach_report(inputs)

        st.success(f"Report ready (generator: {result['llm_provider']}).")

        st.subheader("Coach report")
        st.markdown(result["report_markdown"])

        # lightweight session memory (optional requirement)
        st.session_state.setdefault("coach_history", [])
        st.session_state["coach_history"].append(
            {
                "generated_on": result["generated_on"],
                "goal": inputs.goal,
                "hours_per_week": inputs.hours_per_week,
                "weeks": inputs.weeks,
                "avg": result["diagnosis"]["average_score"],
                "risk": result["diagnosis"]["risk_level"],
                "provider": result["llm_provider"],
                "report": result["report_markdown"],
            }
        )

        md_bytes = result["report_markdown"].encode("utf-8")
        pdf_bytes = report_markdown_to_pdf_bytes("AI Study Coach Report", result["report_markdown"])
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Download report (Markdown)", data=md_bytes, file_name="study_coach_report.md")
        with d2:
            st.download_button("Download report (PDF)", data=pdf_bytes, file_name="study_coach_report.pdf")

        st.divider()
        st.subheader("Retrieved resources (URLs)")
        if result["resources"]:
            for r in result["resources"]:
                st.markdown(f"- **{r.title}**  \n{r.url}")
        else:
            st.info("No resources retrieved (network might be blocked). The plan still works via fallback.")

        st.divider()
        with st.expander("Session memory (previous coach runs)"):
            hist = st.session_state.get("coach_history", [])
            if not hist:
                st.write("No previous runs yet.")
            else:
                st.dataframe(
                    [
                        {
                            "date": h["generated_on"],
                            "avg": h["avg"],
                            "risk": h["risk"],
                            "hours/week": h["hours_per_week"],
                            "weeks": h["weeks"],
                            "provider": h["provider"],
                            "goal": h["goal"],
                        }
                        for h in hist[-20:][::-1]
                    ]
                )

elif menu == "Info":
    st.header("About This App")
    st.write("Built with Streamlit, Pandas, and Scikit-Learn.")
    st.write("This tool helps educators find at-risk students based on demographic data and scores.")
