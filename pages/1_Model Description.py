import streamlit as st 
from home import class_report, test_report

st.set_page_config(layout="wide", page_title="Model Analysis")

st.title("🤖 Model Architecture & Performance Analysis")

st.markdown("---")

st.subheader("The Objective")
st.caption("""
The goal of this model is to determine the **predictive power of metadata** on YouTube Shorts performance. 
We are classifying videos into two categories:
*   **High Engagement:** The video performs better than the median average.
*   **Low Engagement:** The video performs worse than the median average.

**Target Metric:** Engagement Rate = $(Likes + Shares + Comments) / Views$
""")

st.subheader("The Algorithm: Random Forest")
st.caption("""
We utilize a **Random Forest Classifier**, an ensemble learning method that operates by constructing a multitude of decision trees at training time. 
*   **Why Random Forest?** It handles high-dimensional data (like our One-Hot Encoded categories) well and is less prone to overfitting than a single Decision Tree.
*   **Configuration:** We utilized 200 estimators and applied balanced class weights to handle data asymmetries.
""")

st.subheader("Performance Metrics")
col1, col2 = st.columns(2)

with col1:
    st.info("Training Data Performance")
    st.caption("How well the model learned the provided patterns.")
    st.code(class_report)

with col2:
    st.warning("Test Data Performance (Unseen)")
    st.caption("How well the model generalizes to new data.")
    st.code(test_report)

st.subheader("Critical Analysis & Limitations")
st.error("⚠️ The 'Content Gap' Phenomenon")
st.caption("""
**Observation:** We may notice that the model's accuracy is not perfect (not 90%+). 

**Interpretation:** This is a significant finding not a failure. It proves that **metadata alone (Upload hour, Category, Tags, duration) is a weak predictor of virality.** 
A video uploaded at the **perfect time** with the **perfect tags** will still fail if the actual video content is not entertaining.

**Conclusion:** In order to improve this model significantly we would need to introduce **Content Features** (Audio quality, Video resolution, Script sentiment analysis) which are not present in the current dataset. So in conclusion we would need more (qualitative) data to increase the models performance.
""")