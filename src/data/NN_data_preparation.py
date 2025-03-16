import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """
    โหลดข้อมูลจากไฟล์ CSV
    """
    return pd.read_csv(file_path)



def perform_eda(df):
    """
    วิเคราะห์ข้อมูลเบื้องต้น (Exploratory Data Analysis)
    """
    # สรุปข้อมูลทั่วไป
    print("ข้อมูลทั่วไป:")
    print(f"จำนวนข้อมูล: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")
    print("\nตัวอย่างข้อมูล 5 แถวแรก:")
    print(df.head())
    
    # ตรวจสอบประเภทข้อมูล
    print("\nประเภทข้อมูลของแต่ละคอลัมน์:")
    print(df.dtypes)
    
    # ตรวจสอบค่าที่หายไป (Missing Values)
    missing_values = df.isnull().sum()
    print("\nจำนวนค่าที่หายไปในแต่ละคอลัมน์:")
    print(missing_values)
    
    # ตรวจสอบค่าสถิติพื้นฐาน
    print("\nค่าสถิติพื้นฐานของข้อมูลเชิงตัวเลข:")
    print(df.describe())
    
    # สร้างกราฟแสดงการกระจายของราคาบ้าน
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], kde=True)
    plt.title('House Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.savefig('outputs/png/nn/price_distribution.png')
    
    # ตรวจสอบความสัมพันธ์ระหว่างคุณสมบัติต่างๆ กับราคา
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig('outputs/png/nn/correlation_matrix.png')
    
    # สร้างกราฟแสดงความสัมพันธ์ระหว่างพื้นที่กับราคา
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='area', y='price', data=df)
    plt.title('Relationship Between Area and House Price')
    plt.xlabel('Area (sq.m.)')
    plt.ylabel('Price')
    plt.savefig('outputs/png/nn/area_vs_price.png')
    
    # สร้างกราฟแสดงราคาเฉลี่ยตาม neighborhood
    plt.figure(figsize=(12, 6))
    neighborhood_avg_price = df.groupby('neighborhood')['price'].mean().sort_values(ascending=False)
    sns.barplot(x=neighborhood_avg_price.index, y=neighborhood_avg_price.values)
    plt.title('Average Price by Neighborhood')
    plt.xlabel('Neighborhood')
    plt.ylabel('Average Price')
    plt.xticks(rotation=45)
    plt.savefig('outputs/png/nn/avg_price_by_neighborhood.png')
    
    return {
        'missing_values': missing_values,
        'correlation_matrix': correlation_matrix,
        'neighborhood_avg_price': neighborhood_avg_price
    }

def clean_data(df):
    """
    ทำความสะอาดข้อมูล
    """
    # สร้าง DataFrame ใหม่เพื่อไม่ให้กระทบข้อมูลต้นฉบับ
    cleaned_df = df.copy()
    
    # 1. ตรวจสอบและแก้ไขค่าผิดปกติ (Outliers/Anomalies)
    
    # แก้ไขค่า age ที่ติดลบ
    cleaned_df.loc[cleaned_df['age'] < 0, 'age'] = np.nan
    
    # แก้ไขค่า bathrooms ที่มีค่าผิดปกติ (มากกว่า 10 ห้อง)
    cleaned_df.loc[cleaned_df['bathrooms'] > 10, 'bathrooms'] = np.nan
    
    # แก้ไขค่า bedrooms ที่มีค่าผิดปกติ (มากกว่า 10 ห้อง)
    cleaned_df.loc[cleaned_df['bedrooms'] > 10, 'bedrooms'] = np.nan
    
    # 2. แก้ไขการสะกดผิดในคอลัมน์ categorical
    # Map inconsistent spellings to standard forms
    property_type_mapping = {
        'Dettached': 'Detached',
        'Condominium': 'Condo',
        'SemiDetached': 'Semi-Detached',
        'Town-house': 'Townhouse'
    }
    
    neighborhood_mapping = {
        'Down town': 'Downtown',
        'Up-town': 'Uptown',
        'Sub-urban': 'Suburban',
        'Urbann': 'Urban',
        'Rurall': 'Rural'
    }
    
    # Apply the mappings
    cleaned_df['property_type'] = cleaned_df['property_type'].replace(property_type_mapping)
    cleaned_df['neighborhood'] = cleaned_df['neighborhood'].replace(neighborhood_mapping)
    
    # 3. จัดการกับค่าที่หายไป (Missing Values)
    # เราจะทำการจัดการในขั้นตอนการแปลงข้อมูล
    
    return cleaned_df



def transform_data(df):
    """
    แปลงข้อมูลให้พร้อมสำหรับการสร้างโมเดล
    """
    # แยกคอลัมน์ตัวเลขและคอลัมน์ categorical
    numeric_features = ['area', 'bedrooms', 'bathrooms', 'age', 'distance_to_city', 'renovation_year']
    categorical_features = ['property_type', 'neighborhood', 'has_pool']
    
    # สร้าง pipeline สำหรับคอลัมน์ตัวเลข
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # ใช้ค่ามัธยฐานเติมค่าที่หายไป
        ('scaler', StandardScaler())  # ปรับให้มีค่าเฉลี่ย 0 และส่วนเบี่ยงเบนมาตรฐาน 1
    ])
    
    # สร้าง pipeline สำหรับคอลัมน์ categorical
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # ใช้ค่าที่พบบ่อยที่สุดเติมค่าที่หายไป
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # แปลงเป็น one-hot encoding
    ])
    
    # รวม pipeline ด้วย ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def prepare_data_for_model(file_path, test_size=0.2, random_state=42):
    """
    เตรียมข้อมูลทั้งหมดสำหรับการสร้างโมเดล
    """
    # 1. โหลดข้อมูล
    df = load_data(file_path)
    
    # 2. วิเคราะห์ข้อมูลเบื้องต้น
    eda_results = perform_eda(df)
    
    # 3. ทำความสะอาดข้อมูล
    cleaned_df = clean_data(df)
    
    # 4. แยก features และ target
    X = cleaned_df.drop('price', axis=1)
    y = cleaned_df['price']
    
    # 5. แบ่งข้อมูลสำหรับการฝึกฝนและทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # 6. สร้าง preprocessor
    preprocessor = transform_data(cleaned_df)
    
    # 7. แปลงข้อมูลสำหรับการฝึกฝน
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_processed': X_train_processed,
        'X_test_processed': X_test_processed,
        'preprocessor': preprocessor,
        'feature_names': list(X.columns),
        'cleaned_df': cleaned_df,
        'eda_results': eda_results
    }

def save_data_bounds(X):
    """
    บันทึกขอบเขต (min/max) ของแต่ละคอลัมน์ในข้อมูลฝึกสอน
    """
    # สร้าง dictionary เก็บค่า min/max ของแต่ละคอลัมน์
    bounds = {}
    for col in X.columns:
        bounds[col] = {
            'min': X[col].min(),
            'max': X[col].max(),
            'mean': X[col].mean(),
            'std': X[col].std()
        }
    
    # บันทึกลงไฟล์
    joblib.dump(bounds, 'data/models/nn/data_bounds.joblib')
    print(f"บันทึกขอบเขตข้อมูลเรียบร้อยแล้ว")
    return bounds





if __name__ == "__main__":
    # เรียกใช้ฟังก์ชัน prepare_data_for_model เพื่อเตรียมข้อมูล
    data = prepare_data_for_model("data/raw/housing-data.csv")
    print("การเตรียมข้อมูลเสร็จสมบูรณ์!")
    print(f"จำนวนข้อมูลสำหรับการฝึกฝน: {data['X_train'].shape[0]} แถว")
    print(f"จำนวนข้อมูลสำหรับการทดสอบ: {data['X_test'].shape[0]} แถว")
    
    # บันทึกข้อมูลที่เตรียมแล้วสำหรับการใช้งานต่อไป
    import joblib
    joblib.dump(data, 'data/models/nn/prepared_data.joblib')
    print("บันทึกข้อมูลที่เตรียมแล้วในไฟล์ 'prepared_data.joblib'")