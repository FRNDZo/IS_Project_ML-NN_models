import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import sys

# เพิ่มรูทของโปรเจคไปยัง path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def load_data():
    """
    โหลดข้อมูลดิบจากไฟล์ CSV
    """
    try:
        df = pd.read_csv('data/raw/thai_rainfall_data.csv')
        print(f"โหลดข้อมูลสำเร็จ: {len(df)} แถว")
        return df
    except FileNotFoundError:
        print("ไม่พบไฟล์ข้อมูล โปรดตรวจสอบที่อยู่ไฟล์")
        return None

def explore_data(df):
    """
    สำรวจและแสดงข้อมูลเบื้องต้น
    """
    print("\n===== ข้อมูลเบื้องต้น =====")
    print(df.head())
    
    print("\n===== ข้อมูลทางสถิติ =====")
    print(df.describe())
    
    print("\n===== ข้อมูลทั่วไป =====")
    print(df.info())
    
    print("\n===== จำนวนข้อมูลในแต่ละพื้นที่ =====")
    print(df['location'].value_counts())
    
    print("\n===== จำนวนข้อมูลในแต่ละฤดูกาล =====")
    print(df['season'].value_counts())
    
    print("\n===== อัตราส่วนฝนตก/ไม่ตก =====")
    print(df['rainfall'].value_counts(normalize=True) * 100)
    
    # ตรวจสอบค่า Null
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("\n===== ค่า Null ในข้อมูล =====")
        print(null_counts[null_counts > 0])

def check_data_quality(df):
    """
    ตรวจสอบคุณภาพข้อมูลและจัดการปัญหา
    """
    # ตรวจสอบและจัดการค่า outliers สำหรับข้อมูลตัวเลข
    numeric_columns = ['temp_max', 'temp_min', 'temp_avg', 'humidity', 
                       'pressure', 'wind_speed', 'dew_point', 'rain_probability']
    
    print("\n===== ตรวจสอบค่า Outliers =====")
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            print(f"พบ {len(outliers)} แถวที่เป็น outliers ใน {col}")
            
            # แทนที่ outliers ด้วยค่าขอบเขต
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
    
    # ตรวจสอบและจัดการค่า missing
    if df.isnull().sum().sum() > 0:
        print("\n===== จัดการค่า Missing =====")
        # เติมค่า missing ด้วยค่าเฉลี่ยของแต่ละพื้นที่
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col] = df.groupby('location')[col].transform(
                    lambda x: x.fillna(x.mean())
                )
    
    return df

def feature_engineering(df):
    """
    สร้าง features ใหม่
    """
    # แปลงวันที่เป็น datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # สร้าง features จากวันที่
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # สร้าง features จากความสัมพันธ์
    df['temp_range'] = df['temp_max'] - df['temp_min']
    df['temp_dew_diff'] = df['temp_min'] - df['dew_point']
    
    # สร้าง rain_history_sum
    df['rain_history_sum'] = df['rain_1day_ago'] + df['rain_2days_ago'] + df['rain_3days_ago']
    
    # แปลงทิศทางลมเป็นองศา
    direction_to_degrees = {
        'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
        'S': 180, 'SW': 225, 'W': 270, 'NW': 315
    }
    df['wind_direction_deg'] = df['wind_direction'].map(direction_to_degrees)
    
    # แปลงทิศทางลมเป็นค่า sine และ cosine เพื่อรักษาลักษณะวงกลม
    df['wind_direction_sin'] = np.sin(np.radians(df['wind_direction_deg']))
    df['wind_direction_cos'] = np.cos(np.radians(df['wind_direction_deg']))
    
    return df

def prepare_ml_data(df, test_size=0.2, random_state=42):
    """
    เตรียมข้อมูลสำหรับโมเดล ML
    """
    # กำหนด features และ target
    # ตัด 'rain_probability' ออกเพื่อป้องกัน data leakage
    df_ml = df.drop(['rain_probability', 'date', 'wind_direction', 'wind_direction_deg'], axis=1)
    
    # แยก features และ target variable
    X = df_ml.drop('rainfall', axis=1)
    y = df_ml['rainfall']
    
    # แบ่งข้อมูลเป็นชุด train และ test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # กำหนดประเภทของคอลัมน์
    categorical_cols = ['location', 'season', 'month']
    numeric_cols = X.columns.difference(categorical_cols).tolist()
    
    # สร้าง preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ]
    )
    
    # สร้าง preprocessing pipeline
    preprocessing_pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])
    
    # ใช้ pipeline กับข้อมูล
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)
    
    # สร้าง DataFrame ใหม่จากข้อมูลที่ผ่านการประมวลผล
    # (นี่เป็นขั้นตอนเพิ่มเติมเพื่อให้เข้าใจง่ายขึ้น)
    # ได้ feature names จาก preprocessor
    all_features = []
    
    # ได้ชื่อ features จาก numeric columns
    all_features.extend(numeric_cols)
    
    # ได้ชื่อ features จาก categorical columns with one-hot encoding
    ohe = preprocessor.named_transformers_['cat']
    cat_features = []
    for i, col in enumerate(categorical_cols):
        categories = ohe.categories_[i]
        # drop=first ทำให้หนึ่งกลุ่มหายไป
        cat_features.extend([f"{col}_{cat}" for cat in categories[1:]])
    
    all_features = numeric_cols + cat_features
    
    # บันทึกข้อมูลที่ผ่านการประมวลผลแล้วไปยังโฟลเดอร์ processed
    os.makedirs('data/processed/ml', exist_ok=True)
    
    # บันทึก preprocessor
    joblib.dump(preprocessing_pipeline, 'data/processed/ml/preprocessor.pkl')
    
    # บันทึกข้อมูล train และ test
    np.save('data/processed/ml/X_train.npy', X_train_processed)
    np.save('data/processed/ml/X_test.npy', X_test_processed)
    np.save('data/processed/ml/y_train.npy', y_train.values)
    np.save('data/processed/ml/y_test.npy', y_test.values)
    
    # บันทึกรายชื่อ features เพื่อใช้อ้างอิง
    pd.DataFrame({'feature': all_features}).to_csv('data/processed/ml/feature_names.csv', index=False)
    
    # บันทึกข้อมูลดิบของ train และ test
    X_train.reset_index(drop=True).to_csv('data/processed/ml/X_train_raw.csv', index=False)
    X_test.reset_index(drop=True).to_csv('data/processed/ml/X_test_raw.csv', index=False)
    y_train.reset_index(drop=True).to_csv('data/processed/ml/y_train_raw.csv', index=False)
    y_test.reset_index(drop=True).to_csv('data/processed/ml/y_test_raw.csv', index=False)
    
    return X_train, X_test, y_train, y_test, X_train_processed, X_test_processed, all_features

def main():
    """
    ฟังก์ชั่นหลักสำหรับการเตรียมข้อมูล
    """
    print("เริ่มขั้นตอนการเตรียมข้อมูลสำหรับโมเดล Machine Learning...")
    
    # โหลดข้อมูล
    df = load_data()
    if df is None:
        return
    
    # สำรวจข้อมูล
    explore_data(df)
    
    # ตรวจสอบคุณภาพข้อมูล
    df = check_data_quality(df)
    
    # สร้าง features ใหม่
    df = feature_engineering(df)
    
    print("\n===== ข้อมูลหลังการสร้าง Features ใหม่ =====")
    print(df.head())
    
    # เตรียมข้อมูลสำหรับโมเดล
    X_train, X_test, y_train, y_test, X_train_processed, X_test_processed, feature_names = prepare_ml_data(df)
    
    print("\n===== สรุปข้อมูลที่เตรียมแล้ว =====")
    print(f"จำนวนข้อมูล Train: {len(X_train)} แถว")
    print(f"จำนวนข้อมูล Test: {len(X_test)} แถว")
    print(f"จำนวน Features: {len(feature_names)}")
    print(f"สัดส่วนข้อมูลเป้าหมาย Train (ฝนตก): {y_train.mean()*100:.2f}%")
    print(f"สัดส่วนข้อมูลเป้าหมาย Test (ฝนตก): {y_test.mean()*100:.2f}%")
    
    print("\nการเตรียมข้อมูลสำเร็จ! ข้อมูลที่ผ่านการประมวลผลแล้วถูกบันทึกไว้ที่โฟลเดอร์ 'data/processed/ml/'")

if __name__ == "__main__":
    main()