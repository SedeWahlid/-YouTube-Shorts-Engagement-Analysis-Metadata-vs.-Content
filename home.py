import streamlit as st 
import joblib as job
from main import open_data, split_data, random_forest, make_prediction
import plotly.graph_objects as go
import shap 
from streamlit_shap import st_shap

# -- MAIN ALGORITHM --
@st.cache_data
def run_main()-> None:
    df,categories = open_data()
    X_train,y_train,X_valid,y_valid,X_test,y_test = split_data(df)
    model, y_pred , class_report, y_test, test_report = random_forest(X_train,y_train,X_valid,y_valid,X_test, y_test)
    model_data = {
    "model": model,
    "columns": X_train.columns.tolist()
}
    job.dump(model_data, filename="model.pkl")
    print("main loaded")
    return model, y_pred , class_report, y_test, test_report

# getting the categories from the data where categories ==> Dataframe Column
@st.cache_data
def get_categories():
    _,categories =open_data()
    return categories

# drawing the results 
def draw_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Engagement rate Probability in %"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 45], 'color': "red"},
                {'range': [45, 55], 'color': "yellow"},
                {'range': [55, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    st.plotly_chart(fig)

# creating explainer which calculates most influential category influencing the probability
def create_explainer(input_data):
        loaded_model = job.load(filename= "model.pkl")
        explain = shap.TreeExplainer(model=loaded_model["model"], )
        values = explain(input_data, check_additivity= False)
        print(values)
        return values[:,:,1], explain

# -- MAIN FILE RUN -- 
try:
   model, y_pred , class_report, y_test, test_report = run_main()
except Exception as a:
    st.error(f"Main file couldn't run: {a}")
    raise(ProcessLookupError)

 # -- STREAMLIT INTERFACE -- 
st.title("🧠 YouTube Shorts Engagement Analysis: Metadata vs. Content")
st.markdown("---")
st.set_page_config(layout= "wide")
st.info("Note: This model analyzes metadata (Upload hour, Duration, Category, Hashtags). It does not analyze video content quality.", icon="⚠️")

# creating the user inputs
duration_sec = st.slider("Duration in sec",min_value=1,max_value=60)
upload_hour = st.selectbox(label="Upload hour", options= ["morning","afternoon","evening","night"])
category = st.selectbox(label="Category", options= get_categories())
hashtags = st.slider("Amount of Hashtags", min_value= 1, max_value=32)


# --RESULTS -- 
if st.button("Predict Engagement Rate") :
    try:
        prob, df = make_prediction(duration= duration_sec, hashtags= hashtags, hour= upload_hour, category= category)
        st.write(f"Probability of High Engagement: {prob:.2%}")
        try:
            draw_gauge(prob)
        except Exception as g:
            st.error(f"ERROR: Could not draw gauge: {g}")
            raise(ProcessLookupError)
        try: 
            shap_vals, explainer = create_explainer(input_data=df)
            st.subheader("Why did the model make this prediction?")
            st_shap(shap.plots.waterfall(shap_vals[0]), height=500)
            
        except Exception as e:
            st.error(f"Could not create SHAP plot: {e}")
    except Exception as f:
        st.error(f"ERROR: Could not compute probability: {f}")
        raise(ProcessLookupError)