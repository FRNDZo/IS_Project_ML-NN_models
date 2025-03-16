import os
import subprocess
import time

def run_command(command, description):
    """
    รันคำสั่งและแสดงสถานะ
    """
    print(f"[เริ่ม] {description}...")
    start_time = time.time()
    
    # รันคำสั่ง
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # ตรวจสอบผลลัพธ์
    if process.returncode == 0:
        end_time = time.time()
        print(f"[เสร็จสิ้น] {description} (ใช้เวลา {end_time - start_time:.2f} วินาที)")
        return True
    else:
        print(f"[ผิดพลาด] {description}")
        try:
            print(stderr.decode('utf-8'))
        except UnicodeDecodeError:
            print(stderr.decode('utf-8', errors='replace'))
            # หรือใช้: print("ไม่สามารถแสดงข้อความผิดพลาดได้")
        return False

def main():
    print("=" * 50)
    print("เริ่มกระบวนการฝึกฝนโมเดลและรันแอพพลิเคชัน")
    print("=" * 50)
    
    # 1. เตรียมข้อมูล Neural Network
    nn_data_success = run_command("python src/data/NN_data_preparation.py", "การเตรียมข้อมูล Neural Network")
    
    # 2. เตรียมข้อมูล Machine Learning
    ml_data_success = run_command("python src/data/ML_data_preparation.py", "การเตรียมข้อมูล Machine Learning")
    
    # 3. ฝึกฝนโมเดล Neural Network
    if nn_data_success:
        nn_model_success = run_command("python src/models/NN_model_training.py", "การฝึกฝนโมเดล Neural Network")
    else:
        print("ไม่สามารถเตรียมข้อมูล Neural Network ได้ กรุณาตรวจสอบข้อผิดพลาดด้านบน")
        nn_model_success = False
    
    # 4. ฝึกฝนโมเดล Machine Learning
    if ml_data_success:
        ml_model_success = run_command("python src/models/ML_model_training.py", "การฝึกฝนโมเดล Machine Learning")
    else:
        print("ไม่สามารถเตรียมข้อมูล Machine Learning ได้ กรุณาตรวจสอบข้อผิดพลาดด้านบน")
        ml_model_success = False
    
    # 5. รันแอพพลิเคชัน Streamlit หากอย่างน้อยหนึ่งโมเดลสำเร็จ
    if nn_model_success or ml_model_success:
        print("\n[กำลังเริ่ม] แอพพลิเคชัน Streamlit...")
        print("คุณสามารถเข้าถึงแอพได้แล้ว")
        print("กด Ctrl+C เพื่อหยุดการทำงาน")
        subprocess.run("streamlit run Main_Page.py", shell=True)
    else:
        print("ไม่สามารถฝึกฝนโมเดลใดๆ ได้ กรุณาตรวจสอบข้อผิดพลาดด้านบน")

if __name__ == "__main__":
    main()