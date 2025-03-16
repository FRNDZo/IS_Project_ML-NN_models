import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# โหลดโมเดลและข้อมูลที่จำเป็น
@st.cache_resource
def load_models():
    preprocessor = joblib.load('data/models/nn/preprocessor.joblib')
    nn_model = tf.keras.models.load_model('data/models/nn/neural_network_model.h5')
    return preprocessor, nn_model

@st.cache_data
def load_data():
    prepared_data = joblib.load('data/models/nn/prepared_data.joblib')
    return prepared_data

@st.cache_data
def load_normalization_params():
    try:
        # พยายามโหลดพารามิเตอร์การ normalize จากไฟล์
        norm_params = joblib.load('data/models/nn/normalization_params.joblib')
        return norm_params
    except FileNotFoundError:
        # ถ้าไม่พบไฟล์ ใช้ค่าประมาณจากข้อมูลที่มี
        prepared_data = load_data()
        y_train = prepared_data['y_train']
        mean = y_train.mean()
        std = y_train.std()
        # ตรวจสอบขนาดของค่าเฉลี่ย
        if mean < 100:  # ถ้าค่าเฉลี่ยน้อยมาก น่าจะเป็นราคาในหน่วยล้านบาท
            scaling_factor = 1000000  # แปลงเป็นบาท
        else:
            scaling_factor = 1  # ไม่ต้องแปลง
        return {'mean': mean, 'std': std, 'scaling_factor': scaling_factor}
    
def main():
    # ตั้งค่าหน้าเว็บ
    st.title("🏠 ทำนายราคาบ้านด้วย Neural Network")
    st.markdown("---")
    
    # โหลดโมเดลและข้อมูล
    try:
        preprocessor, nn_model = load_models()
        prepared_data = load_data()
        
        # สร้าง sidebar สำหรับเลือกโมเดล
        # st.sidebar.title("ตั้งค่า")
        st.sidebar.info("Neural Network ใช้ Backpropagation และ Adam Optimizer พร้อม ReLU Activation")
        
        # แสดงผลการเปรียบเทียบโมเดล
        if st.sidebar.checkbox("แสดงประสิทธิภาพของโมเดล"):
            st.subheader("ประสิทธิภาพของโมเดล Neural Network")
            try:
                # เปลี่ยนชื่อไฟล์จาก models_comparison.csv เป็น model_comparison_detailed.csv ตามที่บันทึกในไฟล์ฝึกโมเดล
                comparison_df = pd.read_csv("outputs/csv/nn/model_comparison_detailed.csv")
                # แก้ไขการกรองข้อมูล เนื่องจากคอลัมน์เปลี่ยนจาก 'Model' เป็น 'model_name'
                nn_metrics = comparison_df[comparison_df['model_name'] == 'Neural Network']
                st.dataframe(nn_metrics)
            except:
                st.warning("ไม่พบไฟล์ข้อมูลประสิทธิภาพของโมเดล")
        
        # สร้างแท็บสำหรับการทำนายและการวิเคราะห์ข้อมูล
        tab1, tab2 = st.tabs(["ทำนายราคาบ้าน", "ข้อมูลเชิงลึก"])
        
        with tab1:
            st.header("ทำนายราคาบ้านจากคุณสมบัติ")
            
            # สร้างฟอร์มกรอกข้อมูล
            col1, col2, col3 = st.columns(3)
            
            with col1:
                area = st.number_input("พื้นที่ (ตร.ม.)", min_value=50.0, max_value=500.0, value=150.0, step=10.0)
                bedrooms = st.number_input("จำนวนห้องนอน", min_value=1, max_value=10, value=3, step=1)
                bathrooms = st.number_input("จำนวนห้องน้ำ", min_value=1, max_value=5, value=2, step=1)
            
            with col2:
                age = st.number_input("อายุบ้าน (ปี)", min_value=0, max_value=50, value=10, step=1)
                distance_to_city = st.number_input("ระยะห่างจากใจกลางเมือง (กม.)", min_value=0.0, max_value=30.0, value=10.0, step=0.5)
                renovation_year = st.number_input("ปีที่ปรับปรุงล่าสุด", min_value=1990, max_value=2024, value=2015, step=1)
            
            with col3:
                property_type = st.selectbox("ประเภทของบ้าน", ["Detached", "Semi-Detached", "Townhouse", "Condo"])
                neighborhood = st.selectbox("ย่าน", ["Downtown", "Urban", "Suburban", "Rural", "Uptown"])
                has_pool = st.checkbox("มีสระว่ายน้ำ")
            
            # แปลงค่า has_pool เป็น 0 หรือ 1
            has_pool_value = 1 if has_pool else 0
            
            # สร้างปุ่มทำนาย
            predict_button = st.button("ทำนายราคา", type="primary")
            
            if predict_button:
                # สร้างข้อมูลใหม่สำหรับการทำนาย
                new_data = pd.DataFrame({
                    'area': [area],
                    'bedrooms': [bedrooms],
                    'bathrooms': [bathrooms],
                    'age': [age],
                    'distance_to_city': [distance_to_city],
                    'property_type': [property_type],
                    'neighborhood': [neighborhood],
                    'has_pool': [has_pool_value],
                    'renovation_year': [renovation_year]
                })
                
                # แปลงข้อมูลด้วย preprocessor
                new_data_processed = preprocessor.transform(new_data)
                
                # โหลดพารามิเตอร์การ normalize
                norm_params = load_normalization_params()
                scaling_factor = norm_params.get('scaling_factor', 1000000)  # ค่าเริ่มต้นคือ 1 ล้าน
                
                # ทำนายด้วยโมเดล Neural Network
                prediction_raw = nn_model.predict(new_data_processed)[0][0]
                
                # แปลงค่ากลับ (denormalize) ถ้าจำเป็น
                if 'mean' in norm_params and 'std' in norm_params:
                    prediction = prediction_raw * norm_params['std'] + norm_params['mean']
                else:
                    prediction = prediction_raw
                
                # ปรับขนาดตามความเหมาะสม
                scaling_factor = norm_params.get('scaling_factor', 1)  # ใช้ค่าเริ่มต้นคือ 1 ถ้าไม่มี
                if prediction < 1000 and scaling_factor > 1:  # ถ้าราคาน้อยกว่า 1000 บาท และมี scaling factor
                    prediction *= scaling_factor
                
                st.success(f"ราคาบ้านที่ทำนายด้วย Neural Network: ฿{prediction:,.2f}")
                
                # แสดงผลเพิ่มเติมเกี่ยวกับค่าที่ใช้ในการทำนาย
                with st.expander("รายละเอียดการทำนาย"):
                    st.write("คุณสมบัติของบ้านที่ใช้ในการทำนาย:")
                    st.write(new_data)
        
        with tab2:
            st.header("ข้อมูลเชิงลึกเกี่ยวกับราคาบ้าน")
            
            # แสดงภาพจากการวิเคราะห์ข้อมูล
            viz_option = st.selectbox(
                "เลือกการวิเคราะห์ที่ต้องการดู",
                ["การกระจายของราคาบ้าน", "ความสัมพันธ์ระหว่างพื้นที่กับราคา", "ราคาเฉลี่ยตามย่าน", "ความสัมพันธ์ระหว่างคุณสมบัติต่างๆ"]
            )
            
            if viz_option == "การกระจายของราคาบ้าน":
                st.image("outputs/png/nn/price_distribution.png")
                st.write("กราฟแสดงการกระจายของราคาบ้านในชุดข้อมูล")
                
            elif viz_option == "ความสัมพันธ์ระหว่างพื้นที่กับราคา":
                st.image("outputs/png/nn/area_vs_price.png")
                st.write("กราฟแสดงความสัมพันธ์ระหว่างพื้นที่บ้านกับราคา โดยทั่วไปบ้านที่มีพื้นที่มากกว่าจะมีราคาสูงกว่า")
                
            elif viz_option == "ราคาเฉลี่ยตามย่าน":
                st.image("outputs/png/nn/avg_price_by_neighborhood.png")
                st.write("กราฟแสดงราคาเฉลี่ยของบ้านในแต่ละย่าน")
                
            else:  # ความสัมพันธ์ระหว่างคุณสมบัติต่างๆ
                st.image("outputs/png/nn/correlation_matrix.png")
                st.write("เมทริกซ์แสดงความสัมพันธ์ระหว่างคุณสมบัติต่างๆ ของบ้านและราคา")
                st.write("ค่าที่ใกล้ 1 แสดงถึงความสัมพันธ์เชิงบวกที่สูง ค่าที่ใกล้ -1 แสดงถึงความสัมพันธ์เชิงลบที่สูง")
            
            # แสดงข้อมูลเพิ่มเติมเกี่ยวกับโมเดล
            if st.checkbox("แสดงข้อมูลเพิ่มเติมเกี่ยวกับโมเดล"):
                st.subheader("ข้อมูลเพิ่มเติมเกี่ยวกับ Neural Network")
                try:
                    # เปลี่ยนชื่อไฟล์ตามที่บันทึกในไฟล์ฝึกโมเดล
                    st.image("outputs/png/nn/nn_performance_detailed.png")
                    st.write("กราฟซ้ายบน: แสดง Loss ของการฝึกฝนและ Validation ในแต่ละ Epoch")
                    st.write("กราฟขวาบน: แสดงการเปรียบเทียบระหว่างค่าจริงและค่าที่ทำนายได้")
                    st.write("กราฟซ้ายล่าง: แสดงค่า Residuals ของการทำนาย")
                    st.write("กราฟขวาล่าง: แสดงการกระจายของค่าความผิดพลาด")
                except:
                    st.warning("ไม่พบไฟล์รูปภาพแสดงประสิทธิภาพของโมเดล")
                
                try:
                    # เพิ่มการแสดงกราฟเปรียบเทียบโมเดล
                    st.image("outputs/png/nn/model_comparison.png")
                    st.write("กราฟเปรียบเทียบประสิทธิภาพระหว่างโมเดลต่างๆ")
                except:
                    st.warning("ไม่พบไฟล์รูปภาพเปรียบเทียบโมเดล")
    
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ที่จำเป็น กรุณาฝึกฝนโมเดลก่อนรันแอพพลิเคชัน")
        st.info("วิธีการฝึกฝนโมเดล:")
        st.code("python src/data/data_preparation.py", language="bash")
        st.code("python src/models/model_training.py", language="bash")

if __name__ == "__main__":
    main()