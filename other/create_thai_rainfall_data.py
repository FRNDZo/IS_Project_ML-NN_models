import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# กำหนดวันเริ่มต้นและสิ้นสุด
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 2, 2)

# คำนวณจำนวนวัน
delta = end_date - start_date
num_days = delta.days + 1

# รายชื่อพื้นที่
locations = ['Bangkok', 'Chiang_Mai', 'Phuket', 'Khon_Kaen', 'Hat_Yai']

# สร้างรายการว่างเปล่าสำหรับเก็บข้อมูล
data = []

# ค่าพื้นฐานและช่วงสำหรับแต่ละพื้นที่ (อุณหภูมิ, ความชื้น, ความกดอากาศ, dew point)
location_params = {
    'Bangkok': {
        'temp_base': [30, 24, 27],  # [max, min, avg]
        'temp_var': [3, 3, 2],
        'humidity_base': 70,
        'humidity_var': 15,
        'pressure_base': 1010,
        'pressure_var': 5,
        'dew_base': 22,
        'dew_var': 4,
        'wind_speed_base': 8,
        'wind_speed_var': 5,
        'rainy_season': [5, 6, 7, 8, 9, 10],  # พฤษภาคม-ตุลาคม
        'missing_prob': 0.05  # โอกาสที่จะเกิด missing value 5%
    },
    'Chiang_Mai': {
        'temp_base': [33, 20, 27],
        'temp_var': [4, 5, 3],
        'humidity_base': 65,
        'humidity_var': 20,
        'pressure_base': 1000,
        'pressure_var': 6,
        'dew_base': 18,
        'dew_var': 5,
        'wind_speed_base': 6,
        'wind_speed_var': 4,
        'rainy_season': [5, 6, 7, 8, 9],  # พฤษภาคม-กันยายน
        'missing_prob': 0.08  # เชียงใหม่มี missing value เยอะกว่า 8%
    },
    'Phuket': {
        'temp_base': [32, 25, 28],
        'temp_var': [2, 2, 1.5],
        'humidity_base': 80,
        'humidity_var': 10,
        'pressure_base': 1008,
        'pressure_var': 4,
        'dew_base': 24,
        'dew_var': 3,
        'wind_speed_base': 12,
        'wind_speed_var': 6,
        'rainy_season': [5, 6, 7, 8, 9, 10, 11],  # พฤษภาคม-พฤศจิกายน
        'missing_prob': 0.04  # เป็นเมืองท่องเที่ยว มี missing value น้อย 4% 
    },
    'Khon_Kaen': {
        'temp_base': [33, 22, 28],
        'temp_var': [3.5, 4, 2.5],
        'humidity_base': 68,
        'humidity_var': 17,
        'pressure_base': 1005,
        'pressure_var': 5,
        'dew_base': 20,
        'dew_var': 4,
        'wind_speed_base': 7,
        'wind_speed_var': 4,
        'rainy_season': [5, 6, 7, 8, 9],  # พฤษภาคม-กันยายน
        'missing_prob': 0.12  # ขอนแก่นมี missing value เยอะที่สุด 12%
    },
    'Hat_Yai': {
        'temp_base': [33, 24, 29],
        'temp_var': [2.5, 2, 1.5],
        'humidity_base': 78,
        'humidity_var': 12,
        'pressure_base': 1009,
        'pressure_var': 4,
        'dew_base': 23,
        'dew_var': 3,
        'wind_speed_base': 10,
        'wind_speed_var': 5,
        'rainy_season': [5, 6, 9, 10, 11, 12],  # พฤษภาคม-มิถุนายน, กันยายน-ธันวาคม
        'missing_prob': 0.07  # หาดใหญ่มี missing value 7%
    }
}

# ทิศทางลมที่เป็นไปได้
wind_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

# กำหนดคอลัมน์ที่อาจจะมี missing values พร้อมโอกาสของแต่ละคอลัมน์
column_missing_prob = {
    'temp_max': 0.15,      # อุณหภูมิสูงสุดมีโอกาสหายไป 15%
    'temp_min': 0.15,      # อุณหภูมิต่ำสุดมีโอกาสหายไป 15%
    'temp_avg': 0.05,      # อุณหภูมิเฉลี่ยมีโอกาสหายไป 5%
    'humidity': 0.25,      # ความชื้นมีโอกาสหายไป 25%
    'pressure': 0.20,      # ความกดอากาศมีโอกาสหายไป 20%
    'wind_speed': 0.30,    # ความเร็วลมมีโอกาสหายไป 30%
    'wind_direction': 0.30, # ทิศทางลมมีโอกาสหายไป 30%
    'dew_point': 0.40,     # จุดน้ำค้างมีโอกาสหายไป 40%
    'rain_probability': 0.10  # โอกาสฝนตกมีโอกาสหายไป 10%
}

# สร้างข้อมูลสำหรับแต่ละวันและแต่ละพื้นที่
for i in range(num_days):
    current_date = start_date + timedelta(days=i)
    month = current_date.month
    
    for location in locations:
        # กำหนดฤดูกาล (แบ่งเป็น 3 ฤดู: ร้อน หนาว ฝน)
        if month in [3, 4, 5]:
            season = 'Summer'
        elif month in [11, 12, 1, 2]:
            season = 'Winter'
        else:
            season = 'Rainy'
        
        params = location_params[location]
        
        # ปรับค่าตามฤดูกาล
        seasonal_adjustment = {
            'Summer': {'temp': 2, 'humidity': -10, 'rain_prob': -20},
            'Winter': {'temp': -3, 'humidity': -5, 'rain_prob': -15},
            'Rainy': {'temp': -1, 'humidity': 15, 'rain_prob': 30}
        }[season]
        
        # เพิ่มโอกาสฝนตกในช่วงฤดูฝนของแต่ละพื้นที่
        rainy_season_boost = 25 if month in params['rainy_season'] else 0
        
        # อุณหภูมิ (สูงสุด/ต่ำสุด/เฉลี่ย) ปรับตามฤดูกาล
        temp_max = params['temp_base'][0] + seasonal_adjustment['temp'] + np.random.uniform(-params['temp_var'][0], params['temp_var'][0])
        temp_min = params['temp_base'][1] + seasonal_adjustment['temp'] + np.random.uniform(-params['temp_var'][1], params['temp_var'][1])
        # ตรวจสอบว่า min ต้องน้อยกว่า max
        temp_min = min(temp_min, temp_max - 2)
        temp_avg = (temp_max + temp_min) / 2
        
        # ความชื้นสัมพัทธ์ ปรับตามฤดูกาล
        humidity = params['humidity_base'] + seasonal_adjustment['humidity'] + np.random.uniform(-params['humidity_var'], params['humidity_var'])
        humidity = max(40, min(95, humidity))  # จำกัดช่วง 40-95%
        
        # ความกดอากาศ
        pressure = params['pressure_base'] + np.random.uniform(-params['pressure_var'], params['pressure_var'])
        
        # ความเร็วลม/ทิศทางลม
        wind_speed = max(0, params['wind_speed_base'] + np.random.uniform(-params['wind_speed_var'], params['wind_speed_var']))
        wind_direction = random.choice(wind_directions)
        
        # จุดน้ำค้าง (Dew point) - สัมพันธ์กับอุณหภูมิและความชื้น
        dew_point = params['dew_base'] + seasonal_adjustment['temp']/2 + np.random.uniform(-params['dew_var'], params['dew_var'])
        dew_point = min(dew_point, temp_min)  # dew point ไม่ควรสูงกว่า temp_min
        
        # สร้างข้อมูลฝนย้อนหลัง 3 วัน (0 = ไม่ตก, 1 = ตก)
        rain_history = []
        for j in range(3):
            if i-j-1 >= 0:  # ตรวจสอบว่ามีข้อมูลย้อนหลังหรือไม่
                prev_day_index = len(data) - (len(locations) - locations.index(location)) - j*len(locations)
                if prev_day_index >= 0:
                    rain_history.append(data[prev_day_index]['rainfall'])
                else:
                    # ถ้าไม่มีข้อมูล ให้สุ่ม
                    base_prob = 0.3 if month in params['rainy_season'] else 0.1
                    rain_history.append(1 if random.random() < base_prob else 0)
            else:
                # ถ้าไม่มีข้อมูล ให้สุ่ม
                base_prob = 0.3 if month in params['rainy_season'] else 0.1
                rain_history.append(1 if random.random() < base_prob else 0)
        
        # คำนวณโอกาสที่ฝนจะตก
        base_rain_prob = 20 + seasonal_adjustment['rain_prob'] + rainy_season_boost
        
        # ปรับตามปัจจัยต่างๆ
        humidity_factor = (humidity - 60) * 0.8 if humidity > 60 else 0
        pressure_factor = (1010 - pressure) * 0.5 if pressure < 1010 else 0
        dew_factor = (dew_point - 15) * 1.2 if dew_point > 15 else 0
        wind_factor = wind_speed * 0.5 if wind_speed > 10 else 0
        history_factor = sum(rain_history) * 15  # ถ้าฝนตกในวันก่อนหน้า เพิ่มโอกาส
        
        rain_probability = base_rain_prob + humidity_factor + pressure_factor + dew_factor + wind_factor + history_factor
        rain_probability = max(0, min(100, rain_probability))  # จำกัดช่วง 0-100%
        
        # ตัดสินใจว่าฝนตกหรือไม่ (1 = ตก, 0 = ไม่ตก)
        rainfall = 1 if random.random() * 100 < rain_probability else 0
        
        # สร้างข้อมูลพื้นฐาน
        record = {
            'date': current_date.strftime('%Y-%m-%d'),
            'location': location,
            'month': month,
            'season': season,
            'temp_max': round(temp_max, 1),
            'temp_min': round(temp_min, 1),
            'temp_avg': round(temp_avg, 1),
            'humidity': round(humidity, 1),
            'pressure': round(pressure, 1),
            'wind_speed': round(wind_speed, 1),
            'wind_direction': wind_direction,
            'dew_point': round(dew_point, 1),
            'rain_3days_ago': rain_history[2] if len(rain_history) > 2 else 0,  
            'rain_2days_ago': rain_history[1] if len(rain_history) > 1 else 0,
            'rain_1day_ago': rain_history[0],
            'rain_probability': round(rain_probability, 1),
            'rainfall': rainfall  # target variable
        }
        
        # ตรวจสอบหากจะมี missing value ในเรคอร์ดนี้
        # missing value จะเกิดเมื่อ:
        # 1. พื้นที่นี้มีโอกาสเกิด missing value (ตามค่า missing_prob ในพื้นที่)
        # 2. คอลัมน์นี้มีโอกาสเกิด missing value (ตามค่า column_missing_prob)
        if random.random() < params['missing_prob']:
            # เลือกฟิลด์ที่อาจจะมี missing value
            for col, prob in column_missing_prob.items():
                if random.random() < prob:
                    record[col] = np.nan  # กำหนดให้ค่าเป็น NaN (missing)
        
        # เพิ่มข้อมูลลงในรายการ
        data.append(record)

# สร้าง DataFrame
df = pd.DataFrame(data)

# บันทึกเป็นไฟล์ CSV
import os
# สร้างโฟลเดอร์ data/raw หากยังไม่มี
os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/thai_rainfall_data.csv', index=False)

# นับจำนวนค่า missing ในแต่ละคอลัมน์
missing_count = df.isna().sum()
missing_percent = (df.isna().sum() / len(df)) * 100

print(f"สร้างข้อมูลสำเร็จ: {len(df)} แถว")
print(f"ตัวอย่างข้อมูล:\n{df.head()}")

print("\nสรุปจำนวน missing value ในแต่ละคอลัมน์:")
for col, count in missing_count.items():
    if count > 0:
        print(f"{col}: {count} แถว ({missing_percent[col]:.2f}%)")

# สรุปข้อมูลฝนตกในแต่ละพื้นที่
print("\nสรุปข้อมูลฝนตกแยกตามพื้นที่:")
rain_summary = df.groupby('location')['rainfall'].mean() * 100
print(rain_summary)

# สรุปข้อมูล missing value แยกตามพื้นที่
print("\nสรุปข้อมูล missing value แยกตามพื้นที่:")
missing_by_location = df.groupby('location').apply(lambda x: x.isna().sum().sum())
missing_percent_by_location = (missing_by_location / (len(df.columns) * df.groupby('location').size())) * 100
for location, count in missing_by_location.items():
    print(f"{location}: {count} ค่า ({missing_percent_by_location[location]:.2f}%)")

# สรุปข้อมูลตามฤดูกาล
print("\nสรุปข้อมูลฝนตกแยกตามฤดูกาล:")
season_summary = df.groupby('season')['rainfall'].mean() * 100
print(season_summary)