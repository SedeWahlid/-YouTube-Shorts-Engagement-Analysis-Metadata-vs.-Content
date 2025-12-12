# 🧠 YouTube Shorts Engagement Analysis: Metadata vs. Content

### 🚀 Project Overview
This project is an end-to-end Machine Learning pipeline designed to investigate the relationship between video metadata (upload time, hashtags, category, duration) and user engagement on YouTube Shorts. 

Using a **Random Forest Classifier**, the project attempts to predict whether a video will achieve a "High Engagement Rate" based solely on non-content features. The project includes a fully interactive web application built with **Streamlit** to visualize predictions and model explainability.

**Key Finding:** The analysis reveals that while metadata plays a role in discoverability it is a weak predictor of viral success compared to latent content variables (video quality, storytelling, audio etc.), highlighting the "Content Gap" in metadata-based modeling.

### Live Playground
Here is a Live Playground to test the application: https://youtube-shorts-engagement-ai.streamlit.app/

**Note**: It made take a while to load the Playground due to streamlits free tier "sleeping" behaviour.

### 🛠️ Tech Stack
*   **Core:** Python 3.9+
*   **Machine Learning:** Scikit-Learn (Random Forest), Joblib
*   **Explainability:** SHAP (Shapley Additive Explanations)
*   **Data Processing:** Pandas
*   **Visualization:** Plotly, Matplotlib
*   **Web App:** Streamlit

### 📂 Project Structure
```text
├── data/
│   └── youtube_shorts_extended.csv  # Raw dataset
├── main.py                          # ETL, Feature Engineering, and Model Training pipeline
├── home.py                          # Streamlit Frontend (Prediction Interface)
├── pages/
│   └── 1_Model Description.py       # Streamlit Page (Performance Metrics)
├── model.pkl                        # Serialized Model & Column Transformers
└── requirements.txt                 # Dependencies
