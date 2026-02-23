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
menu = st.sidebar.radio("Go to", ["Home", "Batch Data", "Get Recommendations", "Info"])

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

elif menu == "Info":
    st.header("About This App")
    st.write("Built with Streamlit, Pandas, and Scikit-Learn.")
    st.write("This tool helps educators find at-risk students based on demographic data and scores.")
