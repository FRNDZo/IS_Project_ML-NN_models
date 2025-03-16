import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

def main():
    st.title("แนวทางการพัฒนา Neural Network สำหรับทำนายราคาบ้าน")
    st.markdown("---")
    
    # คำอธิบายเบื้องต้น
    st.write("""
    การพัฒนาโมเดล Neural Network เพื่อทำนายราคาบ้านเป็นตัวอย่างของการประยุกต์ใช้ Deep Learning
    กับปัญหาการทำนายราคาอสังหาริมทรัพย์ โดยการใช้ข้อมูลคุณลักษณะของบ้านต่างๆ เช่น พื้นที่ จำนวนห้องนอน 
    ห้องน้ำ ทำเลที่ตั้ง ประเภทของบ้าน และปัจจัยอื่นๆ เพื่อสร้างแบบจำลองที่สามารถทำนายราคาได้อย่างแม่นยำ
    """)
    
    st.header("Dataset ที่นำมาใช้")
    st.write("""
      
      Dataset ได้สร้างโดย https://claude.ai/ 
      และต่อไปจะอธิบาย feature ของ Dataset มีดังนี้       
      &nbsp;&nbsp; - area - พื้นที่ของบ้าน (น่าจะมีหน่วยเป็นตารางเมตร) เช่น 158.4, 102.7, 189.5 ตารางเมตร
         
      &nbsp;&nbsp; - bedrooms - จำนวนห้องนอนในบ้าน โดยส่วนใหญ่มีค่าตั้งแต่ 1-5 ห้อง แต่มีบางรายการที่มีค่าผิดปกติ เช่น 12, 18, 14, 17 ห้อง
               
      &nbsp;&nbsp; - bathrooms - จำนวนห้องน้ำในบ้าน โดยส่วนใหญ่มีค่าตั้งแต่ 1-4 ห้อง แต่มีบางรายการที่มีค่าผิดปกติ เช่น 14, 16, 20 ห้อง และมีค่า missing values
               
      &nbsp;&nbsp; - age - อายุของบ้าน (น่าจะมีหน่วยเป็นปี) โดยมีค่าตั้งแต่ 5-45 ปี และมีค่าผิดปกติบางรายการที่เป็นค่าลบ เช่น -5, -12, -8 ปี
               
      &nbsp;&nbsp; - distance_to_city - ระยะห่างจากตัวเมือง (น่าจะมีหน่วยเป็นกิโลเมตร) เช่น 12.5, 5.3, 18.2 กม. และมี missing values
            
      &nbsp;&nbsp; - property_type - ประเภทของบ้าน มีดังนี้:

      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1.)Detached (บ้านเดี่ยว)
               
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.)Semi-Detached (บ้านแฝด)
               
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.)Condo (คอนโดมิเนียม)
               
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.)Townhouse (ทาวน์เฮาส์)
               

      &nbsp;&nbsp; - neighborhood - ย่านที่อยู่อาศัย มีดังนี้:

      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1.)Downtown (ย่านใจกลางเมือง)
             
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.)Urban (ย่านในเมือง)
             
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.)Suburban (ย่านชานเมือง)
             
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.)Rural (ย่านชนบท)
             
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5.)Uptown (ย่านเมืองชั้นดี)
             

      &nbsp;&nbsp; - has_pool - มีสระว่ายน้ำหรือไม่ (เป็นตัวแปรแบบ binary: 0 = ไม่มี, 1 = มี)
             
      &nbsp;&nbsp; - renovation_year - ปีที่ทำการปรับปรุงบ้านครั้งล่าสุด เช่น 2010, 2015, 2005 และมี missing values
             
      &nbsp;&nbsp; - price - ราคาบ้าน (หน่วยเป็นบาท) เช่น 6,781,245.32 และ 5,123,987.61 บาท
      """)
    
    # การเตรียมข้อมูล
    st.header("1. การเตรียมข้อมูล")
    
    with st.expander("1.1 การโหลดและวิเคราะห์ข้อมูลเบื้องต้น"):
      st.write("""
      กระบวนการเริ่มต้นด้วยการโหลดข้อมูลอสังหาริมทรัพย์จากไฟล์ CSV และทำการวิเคราะห์ข้อมูลเบื้องต้น (EDA)
      เพื่อทำความเข้าใจลักษณะของข้อมูล โดยมีการตรวจสอบ:
      
      - รูปแบบและขนาดของข้อมูล (จำนวนแถวและคอลัมน์)
      - ประเภทข้อมูลของแต่ละคอลัมน์
      - ค่าสถิติพื้นฐาน เช่น ค่าเฉลี่ย มัธยฐาน ส่วนเบี่ยงเบนมาตรฐาน
      - การตรวจสอบค่าที่หายไป (Missing Values)
      - การวิเคราะห์การกระจายของราคาบ้าน
      - ความสัมพันธ์ระหว่างคุณลักษณะต่างๆ กับราคาบ้าน
      """)

    

    st.code("""
    # ตัวอย่างการวิเคราะห์ข้อมูลเบื้องต้น
    print("ข้อมูลทั่วไป:")
    print(f"จำนวนข้อมูล: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")
    
    # ตรวจสอบค่าสถิติพื้นฐาน
    print("ค่าสถิติพื้นฐานของข้อมูลเชิงตัวเลข:")
    print(df.describe())
    """, language="python")
    
    st.write("""
    การวิเคราะห์ความสัมพันธ์ระหว่างคุณลักษณะต่างๆ กับราคาบ้านทำให้เราเข้าใจว่าปัจจัยใดมีผลต่อราคามากที่สุด
    โดยใช้ Correlation Matrix เพื่อหาความสัมพันธ์ระหว่างตัวแปรเชิงตัวเลขทั้งหมด
    """)
    
    with st.expander("1.2 การทำความสะอาดข้อมูล"):
      st.write("""
      ขั้นตอนสำคัญในการเตรียมข้อมูลคือการทำความสะอาดข้อมูล เพื่อจัดการกับข้อมูลที่ไม่สมบูรณ์หรือผิดปกติ:
      
      - การแก้ไขค่าผิดปกติ (Outliers) เช่น อายุบ้านที่เป็นค่าติดลบ
      - การแก้ไขจำนวนห้องนอนหรือห้องน้ำที่มีค่าผิดปกติ (มากกว่า 10 ห้อง)
      - การแก้ไขการสะกดผิดในข้อมูลประเภท categorical เช่น 'Dettached' เป็น 'Detached'
      - การจัดการกับค่าที่หายไป (Missing Values)
      """)
    
    st.code("""
    # ตัวอย่างการทำความสะอาดข้อมูล
    # แก้ไขค่า age ที่ติดลบ
    cleaned_df.loc[cleaned_df['age'] < 0, 'age'] = np.nan
    
    # แก้ไขค่า bathrooms ที่มีค่าผิดปกติ
    cleaned_df.loc[cleaned_df['bathrooms'] > 10, 'bathrooms'] = np.nan
    
    # แก้ไขการสะกดผิดในประเภทบ้าน
    property_type_mapping = {
        'Dettached': 'Detached',
        'Condominium': 'Condo',
        'SemiDetached': 'Semi-Detached',
        'Town-house': 'Townhouse'
    }
    cleaned_df['property_type'] = cleaned_df['property_type'].replace(property_type_mapping)
    """, language="python")
    
    with st.expander("1.3 การแปลงข้อมูล"):
      st.write("""
      ก่อนนำข้อมูลไปใช้กับโมเดล Neural Network เราต้องทำการแปลงข้อมูลให้อยู่ในรูปแบบที่เหมาะสม:
      
      - การจัดการกับคอลัมน์ตัวเลข:
         - ใช้ `SimpleImputer` กับค่ามัธยฐานเพื่อเติมค่าที่หายไป
         - ใช้ `StandardScaler` เพื่อปรับให้มีค่าเฉลี่ย 0 และส่วนเบี่ยงเบนมาตรฐาน 1
         
      - การจัดการกับคอลัมน์ categorical:
         - ใช้ `SimpleImputer` กับค่าที่พบบ่อยที่สุดเพื่อเติมค่าที่หายไป
         - ใช้ `OneHotEncoder` เพื่อแปลงเป็น one-hot encoding
         
      - การแบ่งข้อมูลเป็นชุดฝึกฝน (Training Set) และชุดทดสอบ (Test Set) ในสัดส่วน 80:20
      """)
    
    st.code("""
    # ตัวอย่างการแปลงข้อมูล
    # แยกคอลัมน์ตัวเลขและคอลัมน์ categorical
    numeric_features = ['area', 'bedrooms', 'bathrooms', 'age', 'distance_to_city', 'renovation_year']
    categorical_features = ['property_type', 'neighborhood', 'has_pool']
    
    # สร้าง pipeline สำหรับคอลัมน์ตัวเลข
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # สร้าง pipeline สำหรับคอลัมน์ categorical
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    """, language="python")
    
    # ทฤษฎีของ Neural Network
    st.header("2. ทฤษฎีของ Neural Network")
    
    with st.expander("2.1 พื้นฐานของ Neural Network"):
      st.write("""
      Neural Network (โครงข่ายประสาทเทียม) เป็นแบบจำลองทางคณิตศาสตร์ที่ได้รับแรงบันดาลใจจากการทำงานของสมองมนุษย์
      ประกอบด้วยหน่วยประมวลผลเล็กๆ จำนวนมากที่เรียกว่า "นิวรอน" (neurons) เชื่อมต่อกันเป็นเครือข่าย
      
      โครงสร้างพื้นฐานของ Neural Network ประกอบด้วย:
      
      - **Input Layer**: ชั้นรับข้อมูลเข้า แต่ละนิวรอนรับค่าคุณลักษณะหนึ่งของข้อมูล
      - **Hidden Layers**: ชั้นซ่อนที่ทำการประมวลผลข้อมูล (อาจมีหลายชั้น)
      - **Output Layer**: ชั้นส่งผลลัพธ์ออกมา ในกรณีของการทำนายราคาบ้าน จะมีนิวรอนเดียวที่ให้ค่าราคาที่ทำนาย
      
      แต่ละนิวรอนจะได้รับอินพุตจากนิวรอนในชั้นก่อนหน้า คูณด้วยค่าน้ำหนัก (weights) รวมกับค่าความเอนเอียง (bias)
      จากนั้นส่งผ่านฟังก์ชันกระตุ้น (activation function) เพื่อกำหนดค่าที่ส่งต่อไปยังชั้นถัดไป
      """)
    
    with st.expander("2.2 การเรียนรู้ของ Neural Network"):
      st.write("""
      การเรียนรู้ของ Neural Network เป็นกระบวนการปรับค่าน้ำหนัก (weights) และค่าความเอนเอียง (bias)
      เพื่อให้ผลลัพธ์ที่ได้มีความแม่นยำมากที่สุด โดยมีขั้นตอนดังนี้:
      
      1. **Forward Propagation**: คำนวณผลลัพธ์จากอินพุตโดยส่งผ่านทุกชั้นของโครงข่าย
      
      2. **คำนวณค่าความผิดพลาด (Loss)**: วัดความแตกต่างระหว่างผลลัพธ์ที่ได้กับค่าที่ถูกต้อง
         ในกรณีการทำนายราคาบ้าน เรามักใช้ Mean Squared Error (MSE) เป็นฟังก์ชันความผิดพลาด
         
      3. **Backpropagation**: คำนวณการเปลี่ยนแปลงของค่าความผิดพลาดเมื่อค่าน้ำหนักเปลี่ยนไป
         และส่งค่าความผิดพลาดย้อนกลับไปทุกชั้น
         
      4. **ปรับค่าน้ำหนัก**: ใช้ optimizer (เช่น Adam) เพื่อปรับค่าน้ำหนักในทิศทางที่ทำให้ค่าความผิดพลาดลดลง
      
      กระบวนการนี้ทำซ้ำหลายรอบ (epochs) จนกว่าค่าความผิดพลาดจะลดลงถึงระดับที่น่าพอใจ
      """)
    
    with st.expander("2.3 Activation Functions และ Regularization"):
      st.write("""
      **Activation Functions**:
      
      ฟังก์ชันกระตุ้นทำให้ Neural Network สามารถเรียนรู้ความสัมพันธ์ที่ซับซ้อนได้:
      
      - **ReLU (Rectified Linear Unit)**: ฟังก์ชันที่ใช้ในโมเดลของเรา `f(x) = max(0, x)`
         เป็นฟังก์ชันที่นิยมใช้ในชั้นซ่อน เนื่องจากช่วยแก้ปัญหา vanishing gradient
         
      - **Linear**: ไม่มีการใช้ฟังก์ชันกระตุ้นในชั้น output เนื่องจากเป็นปัญหาการทำนายค่าต่อเนื่อง (regression)
         
      **Regularization**:
      
      เทคนิคที่ช่วยป้องกันปัญหา overfitting ซึ่งเกิดเมื่อโมเดลเรียนรู้เฉพาะข้อมูลฝึกฝนมากเกินไป:
      
      - **Dropout**: เทคนิคที่ใช้ในโมเดลของเรา โดยในระหว่างการฝึกฝน จะสุ่มปิดการทำงานของนิวรอนบางตัว
         เพื่อป้องกันไม่ให้โมเดลพึ่งพานิวรอนตัวใดตัวหนึ่งมากเกินไป
         
      - **Early Stopping**: หยุดการฝึกฝนเมื่อประสิทธิภาพบนชุดข้อมูล validation ไม่ได้ดีขึ้น
         ช่วยป้องกันการ overfit และประหยัดเวลาในการฝึกฝน
      """)
    
    # การสร้างและฝึกฝนโมเดล Neural Network
    st.header("3. การสร้างและฝึกฝนโมเดล Neural Network")
    
    with st.expander("3.1 โครงสร้างของโมเดล"):
      st.write("""
      โมเดล Neural Network ที่พัฒนาขึ้นมีโครงสร้างดังนี้:
      
      1. **Input Layer**: รับข้อมูลคุณลักษณะที่ผ่านการแปลงแล้ว
      
      2. **Hidden Layers**:
         - ชั้นแรก: มี 64 นิวรอน กับ ReLU activation และ Dropout 30%
         - ชั้นที่สอง: มี 48 นิวรอน กับ ReLU activation และ Dropout 20%
         - ชั้นที่สาม: มี 24 นิวรอน กับ ReLU activation
         
      3. **Output Layer**: มีนิวรอนเดียว ไม่มี activation function เนื่องจากเป็นปัญหา regression
      
      การใช้หลายชั้นซ่อนช่วยให้โมเดลสามารถเรียนรู้ความสัมพันธ์ที่ซับซ้อนระหว่างคุณลักษณะต่างๆ และราคาบ้าน
      ขณะที่ Dropout layers ช่วยป้องกันปัญหา overfitting
      """)
    
    st.code("""
    # สร้างโมเดล Neural Network
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dropout(0.3),  # เพิ่ม dropout เพื่อลด overfitting
        Dense(32, activation='relu'), 
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    """, language="python")
    
    with st.expander("3.2 การ Compile และ Train โมเดล"):
      st.write("""
      หลังจากสร้างโครงสร้างโมเดลแล้ว เราต้อง compile และฝึกฝนโมเดล:
      
      1. **Compile**:
         - เลือกใช้ optimizer Adam ซึ่งเป็น adaptive learning rate optimizer ที่มีประสิทธิภาพสูง
         - กำหนด learning rate = 0.0005 ซึ่งค่อนข้างต่ำเพื่อให้การเรียนรู้มีความเสถียรมากขึ้น
         - เลือกใช้ loss function เป็น Mean Squared Error (MSE) เหมาะสำหรับปัญหา regression
         - ติดตามค่า Mean Absolute Error (MAE) เป็น metric เพิ่มเติม
         
      2. **Normalize เป้าหมาย (Target)**:
         - ทำการ normalize ค่าราคาบ้าน โดยการลบด้วยค่าเฉลี่ยและหารด้วยส่วนเบี่ยงเบนมาตรฐาน
         - การ normalize ช่วยให้โมเดลเรียนรู้ได้ดีขึ้น โดยเฉพาะเมื่อค่าเป้าหมายมีช่วงกว้าง
         
      3. **Early Stopping**:
         - ตั้งค่า patience = 15 หมายความว่าจะหยุดการฝึกฝนหากประสิทธิภาพไม่ดีขึ้นใน 15 epochs
         - เลือกติดตาม validation loss และบันทึกโมเดลที่ดีที่สุด
         
      4. **การฝึกฝน**:
         - ใช้จำนวน epochs สูงสุด = 100 (แต่อาจหยุดก่อนเนื่องจาก Early Stopping)
         - ใช้ batch size = 32 ซึ่งเป็นขนาดที่เหมาะสมกับหน่วยความจำและความเร็ว
         - แบ่ง 20% ของข้อมูลฝึกฝนเป็นชุด validation
      """)
    
    st.code("""
    # Compile โมเดล
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    # Normalize ค่าเป้าหมาย
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    # ตั้งค่า Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    # ฝึกฝนโมเดล
    history = model.fit(
        X_train, y_train_norm,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    """, language="python")
    
    with st.expander("3.3 การประเมินโมเดล"):
      st.write("""
      หลังจากฝึกฝนโมเดลเสร็จสิ้น เราทำการประเมินประสิทธิภาพของโมเดลบนชุดข้อมูลทดสอบ:
      
      1. **ทำนายและแปลงกลับ**:
         - ทำนายค่าราคาบ้านจากข้อมูลทดสอบ
         - แปลงค่าที่ทำนายกลับเป็นหน่วยเดิมโดยการคูณด้วยส่วนเบี่ยงเบนมาตรฐานและบวกด้วยค่าเฉลี่ย
         
      2. **คำนวณค่า Metrics**:
         - Mean Squared Error (MSE): วัดค่าความผิดพลาดโดยเฉลี่ยของการทำนาย (ยกกำลังสอง)
         - Root Mean Squared Error (RMSE): รากที่สองของ MSE ให้ค่าในหน่วยเดียวกับราคาบ้าน
         - Mean Absolute Error (MAE): ค่าความผิดพลาดโดยเฉลี่ยแบบสัมบูรณ์
         - R-squared (R²): วัดสัดส่วนของความแปรปรวนในราคาบ้านที่อธิบายได้ด้วยโมเดล
         
      3. **สร้างกราฟแสดงผล**:
         - Loss Curves: แสดงค่า loss ของชุดข้อมูลฝึกฝนและ validation ในแต่ละ epoch
         - Predictions Plot: เปรียบเทียบราคาจริงกับราคาที่ทำนาย
         
      ค่า metrics ที่ได้จะช่วยให้เราประเมินได้ว่าโมเดลสามารถทำนายราคาบ้านได้แม่นยำเพียงใด
      """)
    
    st.code("""
    # ทำนายและแปลงกลับ
    y_pred_norm = model.predict(X_test).flatten()
    y_pred = (y_pred_norm * y_std) + y_mean
    
    # คำนวณ metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # แสดงผล
    print(f"Neural Network: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
    """, language="python")
    
    # ตัวอย่างผลการฝึกฝนและการนำไปใช้
    st.header("4. ผลการฝึกฝนและการนำไปใช้")
    
    with st.expander("4.1 ผลการฝึกฝน"):
      st.write("""
      ผลการฝึกฝนโมเดล Neural Network แสดงให้เห็นว่า:
      
      - **Learning Curves**: ค่า loss ลดลงอย่างต่อเนื่องในช่วงแรก และเริ่มคงที่หลังจากประมาณ 30-40 epochs
      - **ป้องกัน Overfitting**: การใช้ Dropout และ Early Stopping ช่วยให้ค่า validation loss ไม่เพิ่มขึ้น
      
      โมเดล Neural Network มักให้ผลลัพธ์ที่ดีกว่าโมเดลแบบดั้งเดิม (เช่น Linear Regression) เนื่องจาก:
      
      - สามารถเรียนรู้ความสัมพันธ์ที่ไม่เป็นเชิงเส้น (non-linear relationships)
      - สามารถเรียนรู้ปฏิสัมพันธ์ที่ซับซ้อนระหว่างตัวแปรต่างๆ
      - มีความยืดหยุ่นในการปรับโครงสร้างเพื่อให้เหมาะกับข้อมูล
      """)
    
    with st.expander("4.2 การบันทึกและการนำโมเดลไปใช้"):
      st.write("""
      หลังจากฝึกฝนและประเมินโมเดลแล้ว เราบันทึกโมเดลและข้อมูลที่จำเป็นสำหรับการนำไปใช้:
      
      1. **บันทึกโมเดล Neural Network**:
         - บันทึกโมเดลในรูปแบบ .h5 ซึ่งเป็นรูปแบบมาตรฐานของ Keras
         
      2. **บันทึก Preprocessor**:
         - บันทึก preprocessor ที่ใช้แปลงข้อมูลก่อนเข้าโมเดล
         - ประกอบด้วย imputer, scaler และ one-hot encoder
         
      3. **บันทึกพารามิเตอร์การ normalize**:
         - บันทึกค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานที่ใช้ normalize/denormalize ค่าเป้าหมาย
         
      4. **บันทึกผลการเปรียบเทียบ**:
         - บันทึกค่า metrics ต่างๆ ลงในไฟล์ CSV สำหรับการเปรียบเทียบกับโมเดลอื่นๆ
      """)
    
    st.code("""
    # บันทึกโมเดล Neural Network
    nn_results['model'].save('data/models/nn/neural_network_model.h5')
    
    # บันทึก preprocessor
    joblib.dump(prepared_data['preprocessor'], 'data/models/nn/preprocessor.joblib')
    
    # บันทึกพารามิเตอร์การ normalize
    joblib.dump(norm_params, 'data/models/nn/normalization_params.joblib')
    """, language="python")
    
    with st.expander("4.3 การใช้งานโมเดลกับข้อมูลใหม่"):
      st.write("""
      การนำโมเดลที่ฝึกฝนแล้วไปใช้กับข้อมูลใหม่มีขั้นตอนดังนี้:

      1. **โหลดโมเดลและข้อมูลที่จำเป็น**:
      - โหลดโมเดล Neural Network จากไฟล์ .h5
      - โหลด preprocessor ที่บันทึกไว้
      - โหลดค่าพารามิเตอร์การ normalize ที่บันทึกไว้

      2. **เตรียมข้อมูลใหม่**:
      - แปลงข้อมูลใหม่ด้วย preprocessor เดียวกับที่ใช้ในการฝึกฝน
      - ตรวจสอบว่าข้อมูลใหม่มีคอลัมน์ครบถ้วนตามที่โมเดลต้องการ

      3. **ทำนายผลลัพธ์**:
      - ใช้โมเดลทำนายค่าที่ได้รับการ normalize
      - แปลงค่าที่ทำนายกลับเป็นราคาบ้านที่แท้จริงโดยใช้พารามิเตอร์การ normalize

      4. **แสดงผลการทำนาย**:
      - แสดงราคาบ้านที่ทำนายได้
      - สามารถเพิ่มช่วงความเชื่อมั่น (Confidence Interval) โดยใช้เทคนิคเช่น Monte Carlo Dropout
      """)

    st.code("""
    # การโหลดโมเดลและข้อมูลที่จำเป็น
    from keras.models import load_model
    import joblib

    # โหลดโมเดล Neural Network
    model = load_model('data/models/nn/neural_network_model.h5')

    # โหลด preprocessor
    preprocessor = joblib.load('data/models/nn/preprocessor.joblib')

    # โหลดพารามิเตอร์การ normalize
    norm_params = joblib.load('data/models/nn/normalization_params.joblib')
    y_mean, y_std = norm_params['mean'], norm_params['std']

    # เตรียมข้อมูลใหม่
    new_data = pd.DataFrame({
        'area': [150],
        'bedrooms': [3],
        'bathrooms': [2],
        'age': [5],
        'distance_to_city': [8.5],
        'renovation_year': [2018],
        'property_type': ['Detached'],
        'neighborhood': ['Suburban'],
        'has_pool': [False]
    })

    # แปลงข้อมูลด้วย preprocessor
    new_data_processed = preprocessor.transform(new_data)

    # ทำนายและแปลงกลับ
    prediction_norm = model.predict(new_data_processed).flatten()[0]
    prediction = (prediction_norm * y_std) + y_mean

    print(f"ราคาบ้านที่ทำนาย: {prediction:,.2f} บาท")
    """, language="python")

    st.header("🏁 สรุป")

    st.write("""
      จากการพัฒนาโมเดล Neural Network สำหรับการทำนายราคาบ้าน สามารถสรุปได้ดังนี้:
      
      - โมเดล Neural Network ที่พัฒนาขึ้นสามารถเรียนรู้ความสัมพันธ์ที่ซับซ้อนระหว่างคุณลักษณะต่างๆ และราคาบ้าน
      - การใช้หลายชั้นซ่อน (Hidden Layers) พร้อมด้วย Dropout ช่วยให้โมเดลมีความสามารถในการเรียนรู้สูง และป้องกันปัญหา Overfitting
      - การ Normalize ข้อมูลทั้งคุณลักษณะและค่าเป้าหมาย ช่วยให้การฝึกฝนมีประสิทธิภาพมากขึ้น
      - ค่า RMSE และ MAE ที่ได้จากโมเดลอยู่ในระดับที่ยอมรับได้ แสดงให้เห็นว่าโมเดลสามารถทำนายราคาบ้านได้แม่นยำในระดับหนึ่ง
      
      ข้อจำกัดและแนวทางการพัฒนาต่อไป:
      
      - เพิ่มจำนวนข้อมูลฝึกฝนเพื่อให้โมเดลเรียนรู้ได้ดียิ่งขึ้น
      - ทดลองปรับโครงสร้างโมเดล เช่น จำนวนชั้นซ่อน จำนวนนิวรอน และค่า Dropout
      - เพิ่มคุณลักษณะใหม่ เช่น ระยะห่างจากสถานที่สำคัญ คุณภาพโรงเรียนในพื้นที่ อัตราอาชญากรรม
      - ใช้เทคนิค Cross-validation เพื่อประเมินประสิทธิภาพโมเดลได้แม่นยำยิ่งขึ้น
      """)

    st.markdown("---")
    st.header("📚 เอกสารอ้างอิง")
    st.write("""
      ### เอกสารและบทความอ้างอิง
      
      1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. https://www.deeplearningbook.org/
      2. Chollet, F. (2021). Deep Learning with Python (2nd ed.). Manning Publications.
      3. Smith, L. N. (2017). Cyclical Learning Rates for Training Neural Networks. 2017 IEEE Winter Conference on Applications of Computer Vision (WACV). https://arxiv.org/abs/1506.01186
      4. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 15, 1929-1958.
      5. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. 2015 IEEE International Conference on Computer Vision (ICCV). https://arxiv.org/abs/1502.01852
      
      ### ไลบรารีและเครื่องมือที่ใช้
      
      1. TensorFlow/Keras: https://www.tensorflow.org/ - แพลตฟอร์มสำหรับการพัฒนาโมเดล Deep Learning
      2. Scikit-learn: https://scikit-learn.org/ - ไลบรารีสำหรับ Machine Learning ใน Python
      3. Pandas: https://pandas.pydata.org/ - ไลบรารีสำหรับการจัดการและวิเคราะห์ข้อมูล 
      4. Matplotlib: https://matplotlib.org/ - ไลบรารีสำหรับการสร้างกราฟและการแสดงผลข้อมูล
      5. Streamlit: https://streamlit.io/ - เครื่องมือสำหรับการสร้างเว็บแอปพลิเคชันสำหรับการวิเคราะห์ข้อมูลและ Machine Learning
      """)

if __name__ == "__main__":
    main()