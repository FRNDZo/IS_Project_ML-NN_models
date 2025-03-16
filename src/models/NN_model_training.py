import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import tensorflow as tf
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout # type: ignore
from keras.optimizers import Adam # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import KFold
# from keras.wrappers.scikit_learn import KerasRegressor # type: ignore

def create_output_dirs():
    """สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์"""
    os.makedirs('outputs/png/nn', exist_ok=True)
    os.makedirs('outputs/csv/nn', exist_ok=True)
    os.makedirs('data/models/nn', exist_ok=True)

def build_neural_network(input_dim):
    """
    สร้างโมเดล Neural Network สำหรับการทำนายราคาบ้าน
    """
    model = Sequential([
        # Input layer
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        
        # Hidden layers
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        
        # Output layer - Regression problem so no activation
        Dense(1)
    ])
    
    # Compile model with Adam optimizer and MSE loss
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    return model

def calculate_metrics(y_true, y_pred):
    """
    คำนวณ metrics สำหรับประเมินโมเดล regression
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)
    
    # คำนวณ MAPE (Mean Absolute Percentage Error) โดยหลีกเลี่ยงการหารด้วย 0
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # Normalized RMSE
    y_range = y_true.max() - y_true.min()
    nrmse = rmse / y_range if y_range > 0 else 0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'explained_variance': ev,
        'mape': mape,
        'nrmse': nrmse
    }

def plot_model_performance(history, y_test, y_pred, metrics):
    """
    สร้างกราฟแสดงประสิทธิภาพของโมเดล
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Loss curves
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(history['loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    
    # 2. Actual vs Predicted
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(y_test, y_pred, alpha=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_title('Actual vs Predicted')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.text(0.05, 0.95, f"R² = {metrics['r2']:.4f}", transform=ax2.transAxes, fontsize=10)
    
    # 3. Residuals plot
    ax3 = fig.add_subplot(2, 2, 3)
    residuals = y_test - y_pred
    ax3.scatter(y_pred, residuals, alpha=0.5)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_title('Residuals Plot')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Residuals')
    
    # 4. Histogram of errors
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(residuals, bins=30, alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--')
    ax4.set_title('Distribution of Errors')
    ax4.set_xlabel('Error')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('outputs/png/nn/nn_performance_detailed.png')

def train_neural_network(X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """
    ฝึกฝนโมเดล Neural Network และประเมินผล
    """
    print("\n===== ฝึกโมเดล Neural Network =====")
    start_time = time.time()
    
    # Normalize y (สำคัญมากสำหรับ Neural Network)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_norm = (y_train - y_mean) / y_std
    
    # สร้างโมเดล (ใช้โครงสร้างเดิมแต่เพิ่มเติม callbacks)
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )
    
    # ฝึกโมเดล
    history = model.fit(
        X_train, y_train_norm,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # วัดเวลาที่ใช้ในการฝึก
    training_time = time.time() - start_time
    print(f"เวลาที่ใช้ในการฝึก: {training_time:.2f} วินาที")
    
    # ทำนายและแปลงกลับ
    y_test_norm = (y_test - y_mean) / y_std
    y_pred_norm = model.predict(X_test).flatten()
    y_pred = (y_pred_norm * y_std) + y_mean
    
    # คำนวณ metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # แสดงผลลัพธ์
    print("\n===== ผลการประเมินโมเดล Neural Network =====")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # สร้างกราฟ
    plot_model_performance(history.history, y_test, y_pred, metrics)
    
    return {
        'model': model,
        'history': history.history,
        'metrics': metrics,
        'normalization': {'mean': y_mean, 'std': y_std},
        'y_pred': y_pred,
        'training_time': training_time
    }

def compare_with_baseline(X_train, y_train, X_test, y_test, nn_results):
    """
    เปรียบเทียบกับโมเดลอื่น (baseline)
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    
    print("\n===== เปรียบเทียบกับโมเดลอื่น =====")
    
    # เตรียมข้อมูลสำหรับเปรียบเทียบ
    models_results = []
    
    # Linear Regression
    start_time = time.time()
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_time = time.time() - start_time
    lr_metrics = calculate_metrics(y_test, lr_pred)
    lr_metrics['training_time'] = lr_time
    lr_metrics['model_name'] = 'Linear Regression'
    models_results.append(lr_metrics)
    
    # Random Forest
    start_time = time.time()
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_time = time.time() - start_time
    rf_metrics = calculate_metrics(y_test, rf_pred)
    rf_metrics['training_time'] = rf_time
    rf_metrics['model_name'] = 'Random Forest'
    models_results.append(rf_metrics)
    
    # Neural Network (จากผลลัพธ์ที่คำนวณไว้แล้ว)
    nn_metrics = nn_results['metrics']
    nn_metrics['training_time'] = nn_results['training_time']
    nn_metrics['model_name'] = 'Neural Network'
    models_results.append(nn_metrics)
    
    # สร้าง DataFrame
    comparison_df = pd.DataFrame(models_results)
    
    # จัดรูปแบบ DataFrame
    comparison_df = comparison_df[['model_name', 'rmse', 'mae', 'r2', 'explained_variance', 'mape', 'nrmse', 'training_time']]
    comparison_df = comparison_df.sort_values('rmse')
    
    # แสดงและบันทึกผล
    print("\nผลการเปรียบเทียบโมเดล:")
    print(comparison_df)
    comparison_df.to_csv('outputs/csv/nn/model_comparison_detailed.csv', index=False)
    
    # สร้างกราฟเปรียบเทียบ
    plt.figure(figsize=(12, 6))
    
    metrics_to_plot = ['rmse', 'r2', 'explained_variance']
    models = comparison_df['model_name']
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, 3, i+1)
        plt.bar(models, comparison_df[metric])
        plt.title(metric.upper())
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plt.savefig('outputs/png/nn/model_comparison.png')
    
    return comparison_df

def save_normalization_params(y_train):
    """
    คำนวณและบันทึกค่าพารามิเตอร์สำหรับ normalize/denormalize ข้อมูล
    """
    mean = y_train.mean()
    std = y_train.std()
    
    # ตรวจสอบขนาดของข้อมูล
    if mean < 100:  # ถ้าค่าเฉลี่ยน้อย เป็นไปได้ว่าเป็นหน่วยล้านบาท
        scaling_factor = 1000000  # แปลงเป็นบาท
    else:
        scaling_factor = 1  # ไม่ต้องแปลง
    
    norm_params = {
        'mean': mean,
        'std': std,
        'scaling_factor': scaling_factor
    }
    
    # บันทึกพารามิเตอร์
    joblib.dump(norm_params, 'data/models/nn/normalization_params.joblib')
    print("บันทึกพารามิเตอร์การ normalize เรียบร้อยแล้ว")
    
    return norm_params

def train_models():
    """
    ฝึกฝนโมเดล Neural Network และบันทึกผลลัพธ์
    """
    # สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
    create_output_dirs()
    
    # โหลดข้อมูลที่เตรียมไว้แล้ว
    prepared_data = joblib.load('data/models/nn/prepared_data.joblib')
    
    X_train_processed = prepared_data['X_train_processed']
    X_test_processed = prepared_data['X_test_processed']
    y_train = prepared_data['y_train']
    y_test = prepared_data['y_test']
    
    # บันทึกพารามิเตอร์การ normalize
    norm_params = save_normalization_params(y_train)

    # ฝึกฝนโมเดล Neural Network
    nn_results = train_neural_network(X_train_processed, y_train, X_test_processed, y_test)
    
    # เปรียบเทียบกับ baseline models
    comparison_df = compare_with_baseline(X_train_processed, y_train, X_test_processed, y_test, nn_results)
    
    # บันทึกโมเดลสำหรับการใช้งานในอนาคต
    nn_results['model'].save('data/models/nn/neural_network_model.h5')
    
    # บันทึก preprocessor ที่ใช้แปลงข้อมูล
    joblib.dump(prepared_data['preprocessor'], 'data/models/nn/preprocessor.joblib')
    
    print("\nการฝึกฝนและบันทึกโมเดลเสร็จสมบูรณ์!")
    
    return {
        'neural_network': nn_results,
        'comparison': comparison_df
    }

if __name__ == "__main__":
    # ฝึกฝนโมเดล
    results = train_models()