import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Mobile Detail Predictor", layout="centered")
st.title("📱 Phone Detail")

@st.cache_resource
def load_models():
    with open('phone_reg_model.pkl', 'rb') as f:
        reg_model = pickle.load(f)
    with open('phone_clf_model.pkl', 'rb') as f:
        clf_model = pickle.load(f)
    return reg_model, clf_model

try:
    pipeline_reg, pipeline_clf = load_models()
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            brand = st.selectbox("Brand",['OPPO', 'Samsung', 'Apple', 'Vivo', 'Realme', 'Xiaomi'])
        with col2:
            model_name = st.text_input("Model Name", "A53")
        with col3:
            color = st.selectbox("Color", ['Black', 'Blue', 'White', 'Silver', 'Gold'])
        
        submit = st.form_submit_button("Predict All Details")

    if submit:
        input_df = pd.DataFrame({'Brand': [brand], 'Model': [model_name], 'Color': [color]})
        reg_preds = pipeline_reg.predict(input_df)[0]
        clf_preds = pipeline_clf.predict(input_df)[0]

        st.divider()
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.subheader("📊 Price & Rating")
            st.metric("Selling Price", f"₹{reg_preds[3]:,.0f}")
            st.write(f"**Original Price:** ₹{reg_preds[4]:,.0f}")
            st.write(f"**Discount:** {reg_preds[5]:.1f}%")
            st.write(f"**User Rating:** {reg_preds[2]:.1f} ⭐")
        with res_col2:
            st.subheader("⚙️ Hardware & Network")
            st.write(f"**Memory (RAM):** {int(reg_preds[0])} GB")
            st.write(f"**Storage:** {int(reg_preds[1])} GB")
            is_5g = "Yes ✅" if clf_preds[1] == 1 else "No ❌"
            st.write(f"**Is 5G Support:** {is_5g}")
            deal_map = {0: 'Average', 1: 'Good', 2: 'Best'}
            st.write(f"**Deal Status:** {deal_map.get(int(clf_preds[0]), 'N/A')}")
        st.success("Prediction Complete!")
except Exception as e:
    st.error(f"Error: {e}")
