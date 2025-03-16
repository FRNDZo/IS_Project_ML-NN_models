import streamlit as st

def main():
    # ตั้งค่าหน้าเว็บ
    st.set_page_config(
        page_title="หน้าหลัก",
        page_icon="🛖",
        layout="wide"
        
    )

    # หัวข้อหลัก
    
    st.title("🛖 หน้าหลัก")
    st.markdown("---")
    
    # st.write("หน้าหลักสำหรับระบบวิเคราะห์และทำนายข้อมูล")
    # st.write("เนื้อหาหน้าแรกจะถูกเพิ่มในภายหลัง")
    
    # เพิ่มคำอธิบายการใช้งานเว็บไซต์
    st.info("""
    เว็บไซต์นี้ประกอบด้วย 4 หน้า:
    1. Demo Machine Learning
    2. แนวทางการพัฒนา  ทฤษฎีและเทคนิค Machine Learning
    3. Demo การทำนายราคาบ้านด้วย Neural Network
    4. แนวทางการพัฒนา ทฤษฎีและเทคนิค Neural Network สำหรับทำนาย
    
    กรุณาเลือกหน้าที่ต้องการจากเมนูด้านซ้าย
    """)

if __name__ == "__main__":
    main()