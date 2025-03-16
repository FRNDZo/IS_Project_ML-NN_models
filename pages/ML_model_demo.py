import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import sys

# Function to load models
@st.cache_resource
def load_models():
    models = {}
    model_path = 'data/models/ml/'
    
    try:
        models['LogisticRegression'] = joblib.load(os.path.join(model_path, 'LogisticRegression.joblib'))
        models['RandomForest'] = joblib.load(os.path.join(model_path, 'RandomForest.joblib'))
        models['XGBoost'] = joblib.load(os.path.join(model_path, 'XGBoost.joblib'))
        models['NaiveBayes'] = joblib.load(os.path.join(model_path, 'NaiveBayes.joblib'))
        return models
    except FileNotFoundError as e:
        st.error(f"ไม่พบไฟล์โมเดล: {e}")
        st.info("โปรดรันไฟล์ src/models/ml_model_training.py เพื่อสร้างโมเดลก่อน")
        return None

# Function to load preprocessor
@st.cache_resource
def load_preprocessor():
    try:
        return joblib.load('data/processed/ml/preprocessor.pkl')
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ preprocessor")
        st.info("โปรดรันไฟล์ src/data/ml_data_preparation.py เพื่อเตรียมข้อมูลก่อน")
        return None

# Function to load test data
@st.cache_data
def load_test_data():
    try:
        X_test = pd.read_csv('data/processed/ml/X_test_raw.csv')
        y_test = pd.read_csv('data/processed/ml/y_test_raw.csv')
        return X_test, y_test
    except FileNotFoundError:
        st.error("ไม่พบข้อมูลทดสอบ")
        st.info("โปรดรันไฟล์  src/data/ml_data_preparation.py เพื่อเตรียมข้อมูลก่อน")
        return None, None

def main():

    # Title and description
    st.title("🌧️ การพยากรณ์โอกาสที่ฝนจะตกด้วย Machine Learning ")
    st.markdown("---")
    # Main content - Rain Forecast Testing
    st.header("ทดสอบการพยากรณ์ฝน")

    # Load models, preprocessor, and test data
    models = load_models()
    preprocessor = load_preprocessor()
    X_test, y_test = load_test_data()

    if models and preprocessor and X_test is not None:
        # Manual input only
        st.subheader("ป้อนข้อมูลสภาพอากาศ")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location = st.selectbox(
                "พื้นที่",
                ["Bangkok", "Chiang_Mai", "Phuket", "Khon_Kaen", "Hat_Yai"]
            )
            
            month_names = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน", 
               "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]

            selected_month_name = st.selectbox("เดือน", month_names)
            month = month_names.index(selected_month_name) + 1  # แปลงชื่อเดือนเป็นตัวเลข (1-12)

            season_map = {
                (12, 1, 2): "Winter", 
                (3, 4, 5): "Summer",
                (6, 7, 8, 9, 10, 11): "Rainy"
            }

            season = next((s for m_range, s in season_map.items() if month in m_range), "Winter")
            st.write(f"ฤดูกาล: {season}")
            
        with col2:
            temp_max = st.slider("อุณหภูมิสูงสุด (°C)", 20.0, 40.0, 30.0, 0.1)
            temp_min = st.slider("อุณหภูมิต่ำสุด (°C)", 15.0, 30.0, 25.0, 0.1)
            temp_avg = (temp_max + temp_min) / 2
            st.write(f"อุณหภูมิเฉลี่ย: {temp_avg:.1f} °C")
            
            humidity = st.slider("ความชื้น (%)", 30.0, 100.0, 70.0, 0.1)
            pressure = st.slider("ความกดอากาศ (hPa)", 995.0, 1020.0, 1010.0, 0.1)
            
        with col3:
            wind_speed = st.slider("ความเร็วลม (km/h)", 0.0, 20.0, 5.0, 0.1)
            wind_direction = st.selectbox(
                "ทิศทางลม",
                ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            )
            
            dew_point = st.slider("จุดน้ำค้าง (°C)", 15.0, 30.0, 20.0, 0.1)
            
            rain_1day_ago = st.checkbox("ฝนตกเมื่อวานนี้")
            rain_2days_ago = st.checkbox("ฝนตกเมื่อ 2 วันที่แล้ว")
            rain_3days_ago = st.checkbox("ฝนตกเมื่อ 3 วันที่แล้ว")
        
        # Create a dataframe from manual input
        sample_data = pd.DataFrame({
            'location': [location],
            'month': [month],
            'season': [season],
            'temp_max': [temp_max],
            'temp_min': [temp_min],
            'temp_avg': [temp_avg],
            'humidity': [humidity],
            'pressure': [pressure],
            'wind_speed': [wind_speed],
            'wind_direction': [wind_direction],
            'dew_point': [dew_point],
            'rain_3days_ago': [1 if rain_3days_ago else 0],
            'rain_2days_ago': [1 if rain_2days_ago else 0],
            'rain_1day_ago': [1 if rain_1day_ago else 0],
            'rain_history_sum': [int(rain_1day_ago) + int(rain_2days_ago) + int(rain_3days_ago)],
            'temp_range': [temp_max - temp_min],
            'temp_dew_diff': [temp_min - dew_point],
            'is_weekend': [0],  # Placeholder
            'day_of_week': [0],  # Placeholder
            'day_of_year': [month * 30],  # Approximate
            'wind_direction_sin': [0],  # Will be processed by preprocessor
            'wind_direction_cos': [0]   # Will be processed by preprocessor
        })
        
        # Calculate wind direction sine and cosine
        direction_to_degrees = {
            'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
            'S': 180, 'SW': 225, 'W': 270, 'NW': 315
        }
        wind_dir_deg = direction_to_degrees[wind_direction]
        sample_data['wind_direction_sin'] = np.sin(np.radians(wind_dir_deg))
        sample_data['wind_direction_cos'] = np.cos(np.radians(wind_dir_deg))
        
        actual_rainfall = None
        
        # Model selection
        model_name = st.selectbox(
            "เลือกโมเดลที่ต้องการใช้ทำนาย",
            ["LogisticRegression", "RandomForest", "XGBoost", "NaiveBayes", "ทุกโมเดล"]
        )
        
        if st.button("พยากรณ์"):
            # Preprocess the data
            # Need to drop wind_direction as it's not used by preprocessor
            sample_data_processed = preprocessor.transform(sample_data.drop('wind_direction', axis=1))
            
            if model_name == "ทุกโมเดล":
                # Predict with all models
                st.subheader("ผลการพยากรณ์จากทุกโมเดล")
                
                results = []
                
                for name, model in models.items():
                    prob = model.predict_proba(sample_data_processed)[0, 1]
                    prediction = "ฝนตก" if prob > 0.5 else "ฝนไม่ตก"
                    results.append({
                        "โมเดล": name,
                        "ความน่าจะเป็นที่ฝนจะตก": f"{prob:.2%}",
                        "ผลการพยากรณ์": prediction
                    })
                
                # Show results in a table
                results_df = pd.DataFrame(results)
                st.table(results_df)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Get probabilities for bar chart
                probs = [float(r["ความน่าจะเป็นที่ฝนจะตก"].strip("%"))/100 for r in results]
                model_names = [r["โมเดล"] for r in results]
                
                bars = ax.bar(model_names, probs, color=["skyblue" if p <= 0.5 else "lightcoral" for p in probs])
                
                # Add labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.2%}', ha='center', va='bottom')
                
                plt.axhline(y=0.5, color='gray', linestyle='--')
                plt.ylabel("Probability of Rain")
                plt.ylim(0, 1.1)
                plt.title("Comparison of Forecasts from All Models")
                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                # Predict with selected model
                model = models[model_name]
                prediction = model.predict(sample_data_processed)[0]
                probability = model.predict_proba(sample_data_processed)[0, 1]
                
                # Display results
                st.subheader("ผลการพยากรณ์")
                
                # Create columns for visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display prediction
                    st.metric(
                        "ความน่าจะเป็นที่ฝนจะตก",
                        f"{probability:.2%}"
                    )
                    
                    result_text = "ฝนตก" if prediction == 1 else "ฝนไม่ตก"
                    result_color = "lightcoral" if prediction == 1 else "skyblue"
                    
                    st.markdown(f"""
                    <div style="background-color: {result_color}; padding: 20px; border-radius: 10px; text-align: center;">
                        <h2>ผลการพยากรณ์: {result_text}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Create probability gauge
                    fig, ax = plt.subplots(figsize=(6, 4))
                    
                    # Create a simple gauge chart
                    ax.add_patch(plt.Rectangle((0, 0), 1, 0.2, color='skyblue', alpha=0.8))
                    ax.add_patch(plt.Rectangle((0.5, 0), 0.5, 0.2, color='lightcoral', alpha=0.8))
                    
                    # Add probability marker
                    marker_pos = min(max(probability, 0), 1)
                    ax.arrow(marker_pos, 0.3, 0, -0.05, head_width=0.03, head_length=0.05, fc='black', ec='black')
                    
                    # Add text
                    ax.text(0.25, 0.1, "it's not raining.", ha='center', va='center')
                    ax.text(0.75, 0.1, "It's raining.", ha='center', va='center')
                    ax.text(marker_pos, 0.4, f"{probability:.2%}", ha='center', va='center', fontweight='bold')
                    
                    # Customize plot
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 0.5)
                    ax.set_aspect('equal')
                    ax.axis('off')
                    plt.tight_layout()
                    st.pyplot(fig)
    else:
        st.info("ไม่พบโมเดลหรือข้อมูลที่จำเป็น โปรดรันไฟล์เตรียมข้อมูลและฝึกโมเดลก่อน")


if __name__ == "__main__":
    main()



