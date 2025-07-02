import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Mental Health in Tech", layout="wide")

# -----------------------------
# Loading data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("survey.csv")
    df.drop(columns=['state','country','comments','Timestamp'],inplace=True, errors='ignore')

    # Data cleaning
    df = df[df["Age"].between(18, 100)]
    df['Gender'] = df['Gender'].replace(
        ['Male ', 'male', 'M', 'm', 'Cis Male', 'Man', 'cis male', 'Mail', 'Male (CIS)', 'Cis Man', 'msle', 'Malr', 'Mal', 'maile', 'Make'],
        'Male')
    df['Gender'] = df['Gender'].replace(
        ['Female ', 'female', 'F', 'f', 'Woman', 'femail', 'Cis Female', 'cis-female/femme', 'Femake', 'Female (cis)', 'woman'],
        'Female')
    df['Gender'] = df['Gender'].apply(lambda x: 'Other' if x not in ['Male', 'Female'] else x)
    df['work_interfere'] = df['work_interfere'].fillna('Don\'t know')
    df['self_employed'] = df['self_employed'].fillna('No')


    return df

df = load_data()

# -----------------------------
# TITLE
# -----------------------------
st.title("Mental Health in Tech – EDA Dashboard")

st.markdown(
    "This dashboard presents insights from the 2014 OSMI survey on mental health in the tech industry, "
    "analyzing personal factors, work environment, and behavioral responses to mental health challenges."
)

# -----------------------------
# Global Filters
# -----------------------------
st.sidebar.header("🔎 Filters")
gender_filter = st.sidebar.multiselect("Filter by Gender", df["Gender"].unique(), default=df["Gender"].unique())

df_filtered = df[df["Gender"].isin(gender_filter)]


# -----------------------------
# SECTION 1 – Personal info
# -----------------------------
st.header("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Age Distribution")
    fig = px.histogram(df_filtered, x="Age", nbins=20, color="treatment")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Gender Breakdown")
    gender_counts = df_filtered["Gender"].value_counts(normalize=True).mul(100).round(1)
    fig2 = px.pie(names=gender_counts.index, values=gender_counts.values, title="Gender Distribution")
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# SECTION 2 – Work environment
# -----------------------------
st.header("🏢 Work Environment")

env_cols = ["self_employed", "remote_work", "tech_company", "benefits", "care_options", "wellness_program", "anonymity"]
for col in env_cols:
    st.subheader(f"{col.replace('_', ' ').capitalize()} vs Treatment")
    fig = px.histogram(df_filtered, x=col, color="treatment", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# SECTION 3 – Work interference
# -----------------------------
st.header("📌 Work Interference and Seeking Help")

# Distribution globale
col3, col4 = st.columns(2)

with col3:
    st.subheader("Work Interference Perception")
    perc = df_filtered['work_interfere'].value_counts(normalize=True).rename_axis('work_interfere').reset_index(name='Percentage')
    fig3 = px.bar(perc.round(2), x='work_interfere', y='Percentage', text='Percentage')
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Work Interference vs Treatment")
    grouped = df_filtered.groupby(['work_interfere', 'treatment']).size().reset_index(name='count')
    grouped['percentage'] = grouped.groupby('work_interfere')['count'].transform(lambda x: x / x.sum() * 100)
    fig4 = px.bar(grouped.round(2), x='work_interfere', y='percentage', color='treatment',
                  text='percentage', barmode='group')
    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# SECTION 4A – Correlation with Treatment (Bar Chart)
# -----------------------------
st.header("📈 Top Features Correlated with Seeking Treatment")

st.markdown(
    """
    This chart shows the strength of linear correlation (Pearson) between each variable and whether the person sought treatment.  
    """
)

from sklearn.preprocessing import LabelEncoder

# Encode categorical vars
df_corr = df_filtered.copy()
df_corr = df_corr.dropna(subset=["treatment"])
le = LabelEncoder()

for col in df_corr.select_dtypes(include='object').columns:
    df_corr[col] = le.fit_transform(df_corr[col].astype(str))

# Calculate correlations with treatment
correlations = df_corr.corr(numeric_only=True)["treatment"].drop("treatment").sort_values(key=abs, ascending=False)
top_corr = correlations.head(10)

# Plot bar chart
fig_corr_bar = px.bar(
    top_corr,
    x=top_corr.index,
    y=top_corr.values,
    labels={"x": "Variable", "y": "Correlation"},
    title="Top 10 Features Correlated with Treatment",
    color=top_corr.values,
    color_continuous_scale="RdBu",
)
st.plotly_chart(fig_corr_bar, use_container_width=True)

st.markdown("📌 **Strongest correlations** include `work_interfere`, `mental_health_consequence`, and `anonymity`.")

# -----------------------------
# SECTION 4B – HR Practices Relationship Map
# -----------------------------
st.header("🏢 HR Practices that Often Go Together")

st.markdown(
    """
    This heatmap shows how HR-related features correlate with each other.  
    It helps understand whether companies that offer **one type of support** (like mental health benefits) also tend to offer others (like wellness programs or anonymity).
    """
)

hr_vars = ['benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave']

df_hr = df_filtered[hr_vars].dropna()
df_hr_encoded = df_hr.apply(lambda col: le.fit_transform(col.astype(str)) if col.dtype == 'object' else col)

hr_corr = df_hr_encoded.corr()

fig_hr, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(hr_corr, annot=True, cmap='rocket', vmin=0, vmax=1, linewidths=0.5)
st.pyplot(fig_hr)

st.markdown("✅ Higher values indicate that **HR policies tend to be implemented together** in the same organizations.")
