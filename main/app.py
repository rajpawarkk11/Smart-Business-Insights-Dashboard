# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
import joblib
import os

# ---------------- Try importing st_aggrid ----------------
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    aggrid_available = True
except ModuleNotFoundError:
    aggrid_available = False

# ---------------- Page Config ----------------
st.set_page_config(page_title="Smart Business Insights Dashboard", layout="wide")

# ---------------- Video Background ----------------
st.markdown("""
<style>
body {margin:0;padding:0;overflow-x:hidden;}
#video-bg {position: fixed; top:0; left:0; width:100vw; height:100vh; object-fit: cover; z-index: -999; opacity: 0.25;}
.stApp > div:first-child {background: rgba(0,0,0,0.45) !important; backdrop-filter: blur(10px); border-radius:12px; padding:12px;}
.stButton>button {background-color:#1f77b4; color:white; border-radius:8px; padding:8px 16px; transition: all 0.3s ease;}
.stButton>button:hover {background-color:#ff7f0e; transform: scale(1.05);}
.stButton>button:active {transform: scale(0.95);}
</style>

<video id="video-bg" autoplay muted loop>
<source src="https://www.videezy.com/wp-content/uploads/2019/11/VID_20191119_143455.mp4" type="video/mp4">
</video>
""", unsafe_allow_html=True)

st.title("🚀Smart Business Insights Dashboard")

# ---------------- Sidebar ----------------
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", ["Dataset", "Visualization", "ML", "Prediction", "KPIs"])
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

df = None
numeric_cols = []
categorical_cols = []

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# ---------------- Dataset ----------------
if section == "Dataset":
    st.header("1️⃣ Dataset Preview & Info")
    if df is not None:
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.write("Missing values:", df.isna().sum().sum())
        with st.expander("Preview Dataset"):
            if aggrid_available:
                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_default_column(editable=False, groupable=True, filterable=True, sortable=True)
                AgGrid(df, gridOptions=gb.build(), theme='material', height=400)
            else:
                st.dataframe(df.head())
        with st.expander("Column Types"):
            st.write("Numeric Columns:", numeric_cols)
            st.write("Categorical Columns:", categorical_cols)
    else:
        st.warning("Please upload a CSV file from the sidebar.")

# ---------------- Visualization ----------------
elif section == "Visualization":
    st.header("2️⃣ Visualizations")
    if df is not None:
        plot_type = st.selectbox("Plot Type", ["Line", "Bar", "Scatter"])
        x_axis = st.selectbox("X Axis", df.columns)
        y_axis = st.selectbox("Y Axis (optional)", [None]+list(df.columns))
        filter_col = st.selectbox("Filter Column (optional)", [None]+list(df.columns))
        filter_val = None
        if filter_col:
            filter_val = st.text_input(f"Filter {filter_col} by value")
        if st.button("Generate Plot", key="plot_btn"):
            plot_df = df.copy()
            if filter_col and filter_val:
                plot_df = plot_df[plot_df[filter_col].astype(str).str.contains(filter_val)]
            template = 'plotly_dark'
            if plot_type=="Line":
                fig = px.line(plot_df, x=x_axis, y=y_axis if y_axis else x_axis, template=template)
            elif plot_type=="Bar":
                fig = px.bar(plot_df, x=x_axis, y=y_axis if y_axis else x_axis, template=template)
            else:
                fig = px.scatter(plot_df, x=x_axis, y=y_axis if y_axis else x_axis, template=template)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Upload a CSV to visualize.")

# ---------------- ML ----------------
elif section == "ML":
    st.header("3️⃣ ML: Train Model")
    if df is not None:
        target_column = st.selectbox("Select Target Column", df.columns)
        # Auto detect ML type
        ml_task = "Regression" if np.issubdtype(df[target_column].dtype, np.number) else "Classification"
        st.markdown(f"**Detected ML Task:** {ml_task}")

        feature_columns = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target_column], default=[col for col in df.columns if col != target_column])

        if st.button("Train Model", key="train_btn"):
            try:
                if ml_task=="Regression":
                    X = df[feature_columns].select_dtypes(include=[np.number])
                    y = df[target_column]
                    if X.shape[1]==0:
                        st.warning("No numeric features for regression. Select numeric columns.")
                    else:
                        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
                        model = LinearRegression()
                        model.fit(X_train,y_train)
                        preds = model.predict(X_test)
                        st.success("Regression model trained!")
                        st.metric("R² Score", f"{r2_score(y_test,preds):.2f}")
                        joblib.dump((model,X.columns.tolist()), "model.pkl")
                else:
                    X = pd.get_dummies(df[feature_columns])
                    y = df[target_column].astype(str)
                    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
                    model = RandomForestClassifier()
                    model.fit(X_train,y_train)
                    preds = model.predict(X_test)
                    st.success("Classification model trained!")
                    st.metric("Accuracy", f"{accuracy_score(y_test,preds):.2f}")
                    joblib.dump((model,X.columns.tolist()), "model.pkl")
            except Exception as e:
                st.error(f"Error training model: {e}")
    else:
        st.warning("Upload a CSV to train ML model.")

# ---------------- Prediction ----------------
elif section == "Prediction":
    st.header("4️⃣ Predict on New Data")
    if df is not None:
        new_file = st.file_uploader("Upload new CSV for predictions", type=["csv"], key="newdata")
        if new_file is not None:
            if not os.path.exists("model.pkl"):
                st.warning("Train a model first.")
            else:
                try:
                    model, feature_columns = joblib.load("model.pkl")
                    new_df = pd.read_csv(new_file)
                    X_new = pd.get_dummies(new_df)
                    # Foolproof: add missing columns
                    for col in feature_columns:
                        if col not in X_new.columns:
                            X_new[col] = 0
                    X_new = X_new[feature_columns]
                    preds = model.predict(X_new)
                    new_df["Predictions"] = preds
                    st.dataframe(new_df)
                    csv = new_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Predictions CSV", data=csv, file_name="predictions.csv")
                except Exception as e:
                    st.error(f"Error predicting: {e}")
    else:
        st.warning("Upload a CSV to predict.")

# ---------------- KPIs ----------------
elif section == "KPIs":
    st.header("5️⃣ Key Metrics")
    if df is not None:
        st.metric("Total Rows", df.shape[0])
        st.metric("Total Columns", df.shape[1])
        st.metric("Missing Values", df.isna().sum().sum())
        if "units_sold" in df.columns and "price" in df.columns:
            total_sales = (df["units_sold"]*df["price"]).sum()
            st.metric("Total Revenue", f"${total_sales:,.2f}")
    else:
        st.warning("Upload a CSV to calculate KPIs.")


