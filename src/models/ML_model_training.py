import numpy as np
import pandas as pd
import joblib
import os
import sys
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, classification_report, roc_curve, 
                            roc_auc_score, precision_recall_curve, average_precision_score)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import xgboost as xgb

# เพิ่มรูทของโปรเจคไปยัง path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def load_processed_data():
    """
    Load processed data and handle missing values
    """
    try:
        X_train = np.load('data/processed/ml/X_train.npy')
        X_test = np.load('data/processed/ml/X_test.npy')
        y_train = np.load('data/processed/ml/y_train.npy')
        y_test = np.load('data/processed/ml/y_test.npy')
        
        # Load feature names
        feature_names = pd.read_csv('data/processed/ml/feature_names.csv')['feature'].tolist()
        
        print("Data loaded successfully")
        
        # Check for NaN values
        train_nan_mask = np.isnan(X_train).any(axis=1)
        test_nan_mask = np.isnan(X_test).any(axis=1)
        
        if train_nan_mask.sum() > 0 or test_nan_mask.sum() > 0:
            print(f"Dropping {train_nan_mask.sum()} training samples with NaN values")
            print(f"Dropping {test_nan_mask.sum()} test samples with NaN values")
            
            X_train = X_train[~train_nan_mask]
            y_train = y_train[~train_nan_mask]
            X_test = X_test[~test_nan_mask]
            y_test = y_test[~test_nan_mask]
        
        return X_train, X_test, y_train, y_test, feature_names
    
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please run ml_data_preparation.py first to prepare the data")
        return None, None, None, None, None

def train_logistic_regression(X_train, y_train):
    """
    ฝึกโมเดล Logistic Regression
    """
    print("\n===== ฝึกโมเดล Logistic Regression =====")
    
    # Define parameter grid with valid combinations
    param_grid = [
        # For 'liblinear' solver
        {
            'solver': ['liblinear'],
            'C': np.logspace(-3, 3, 7),
            'penalty': ['l1', 'l2'],
            'class_weight': ['balanced', None]
        },
        # For 'lbfgs' solver
        {
            'solver': ['lbfgs'],
            'C': np.logspace(-3, 3, 7),
            'penalty': ['l2', None],
            'class_weight': ['balanced', None]
        },
        # For 'saga' solver
        {
            'solver': ['saga'],
            'C': np.logspace(-3, 3, 7),
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'class_weight': ['balanced', None],
            'l1_ratio': [0.2, 0.5, 0.8]  # Only needed for elasticnet
        }
    ]
    
    # สร้างโมเดล base
    lr_base = LogisticRegression(random_state=42, max_iter=1000)
    
    # สร้าง RandomizedSearchCV
    lr_search = RandomizedSearchCV(
        lr_base, param_grid, n_iter=20, cv=5, 
        scoring='f1', random_state=42, n_jobs=-1
    )
    
    # ฝึกโมเดล
    start_time = time.time()
    lr_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # แสดงผลลัพธ์
    print(f"เวลาที่ใช้ในการฝึก: {training_time:.2f} วินาที")
    print(f"ค่าพารามิเตอร์ที่ดีที่สุด: {lr_search.best_params_}")
    print(f"คะแนน Cross-validation (F1): {lr_search.best_score_:.4f}")
    
    # ได้โมเดลที่ดีที่สุด
    lr_model = lr_search.best_estimator_
    
    return lr_model

def train_random_forest(X_train, y_train):
    """
    ฝึกโมเดล Random Forest
    """
    print("\n===== ฝึกโมเดล Random Forest =====")
    
    # กำหนดพารามิเตอร์สำหรับ RandomizedSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # สร้างโมเดล base
    rf_base = RandomForestClassifier(random_state=42)
    
    # สร้าง RandomizedSearchCV
    rf_search = RandomizedSearchCV(
        rf_base, param_grid, n_iter=20, cv=5, 
        scoring='f1', random_state=42, n_jobs=-1
    )
    
    # ฝึกโมเดล
    start_time = time.time()
    rf_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # แสดงผลลัพธ์
    print(f"เวลาที่ใช้ในการฝึก: {training_time:.2f} วินาที")  
    print(f"ค่าพารามิเตอร์ที่ดีที่สุด: {rf_search.best_params_}")
    print(f"คะแนน Cross-validation (F1): {rf_search.best_score_:.4f}")
    
    # ได้โมเดลที่ดีที่สุด
    rf_model = rf_search.best_estimator_
    
    return rf_model

def train_gradient_boosting(X_train, y_train):
    """
    ฝึกโมเดล Gradient Boosting (XGBoost)
    """
    print("\n===== ฝึกโมเดล XGBoost =====")
    
    # กำหนดพารามิเตอร์สำหรับ RandomizedSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3],
        'scale_pos_weight': [1, 3, 5]
    }
    
    # สร้างโมเดล base
    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # สร้าง RandomizedSearchCV
    xgb_search = RandomizedSearchCV(
        xgb_base, param_grid, n_iter=20, cv=5, 
        scoring='f1', random_state=42, n_jobs=-1
    )
    
    # ฝึกโมเดล
    start_time = time.time()
    xgb_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # แสดงผลลัพธ์
    print(f"เวลาที่ใช้ในการฝึก: {training_time:.2f} วินาที")
    print(f"ค่าพารามิเตอร์ที่ดีที่สุด: {xgb_search.best_params_}")
    print(f"คะแนน Cross-validation (F1): {xgb_search.best_score_:.4f}")
    
    # ได้โมเดลที่ดีที่สุด
    xgb_model = xgb_search.best_estimator_
    
    return xgb_model

def train_naive_bayes(X_train, y_train):
    """
    ฝึกโมเดล Naive Bayes
    """
    print("\n===== ฝึกโมเดล Naive Bayes =====")
    
    # กำหนดพารามิเตอร์สำหรับ RandomizedSearchCV
    param_grid = {
        'var_smoothing': np.logspace(-10, -8, 20)
    }
    
    # สร้างโมเดล base
    nb_base = GaussianNB()
    
    # สร้าง RandomizedSearchCV
    nb_search = RandomizedSearchCV(
        nb_base, param_grid, n_iter=10, cv=5, 
        scoring='f1', random_state=42, n_jobs=-1
    )
    
    # ฝึกโมเดล
    start_time = time.time()
    nb_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # แสดงผลลัพธ์
    print(f"เวลาที่ใช้ในการฝึก: {training_time:.2f} วินาที")
    print(f"ค่าพารามิเตอร์ที่ดีที่สุด: {nb_search.best_params_}")
    print(f"คะแนน Cross-validation (F1): {nb_search.best_score_:.4f}")
    
    # ได้โมเดลที่ดีที่สุด
    nb_model = nb_search.best_estimator_
    
    return nb_model

def evaluate_model(model, X_test, y_test, model_name, feature_names=None):
    """
    ประเมินประสิทธิภาพของโมเดล
    """
    print(f"\n===== ประเมินโมเดล {model_name} =====")
    
    # ทำนายบนชุดข้อมูล test
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # คำนวณ metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    
    # แสดงผลลัพธ์
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
    os.makedirs('outputs/png/ml', exist_ok=True)
    os.makedirs('outputs/csv/ml', exist_ok=True)
    
    # บันทึกผลการทำนายไปยังไฟล์ CSV
    results_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob
    })
    results_df.to_csv(f'outputs/csv/ml/{model_name}_predictions.csv', index=False)
    
    # วาด ROC curve
    plt.figure(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'outputs/png/ml/{model_name}_roc_curve.png')
    
    # วาด Precision-Recall curve
    plt.figure(figsize=(10, 6))
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall_curve, precision_curve, lw=2, label=f'PR curve (AP = {ap:.4f})')
    plt.axhline(y=sum(y_test)/len(y_test), color='r', linestyle='--', label=f'Baseline (AP = {sum(y_test)/len(y_test):.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.savefig(f'outputs/png/ml/{model_name}_pr_curve.png')
    
    # แสดง feature importances สำหรับโมเดลที่สนับสนุน
    if model_name in ['RandomForest', 'XGBoost'] and feature_names is not None:
        plt.figure(figsize=(12, 8))
        
        if model_name == 'RandomForest':
            importances = model.feature_importances_
        elif model_name == 'XGBoost':
            importances = model.feature_importances_
        
        # สร้าง DataFrame จาก feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # บันทึก feature importances ลงไฟล์ CSV
        feature_importance_df.to_csv(f'outputs/csv/ml/{model_name}_feature_importance.csv', index=False)
        
        # แสดง top N features
        top_n = 20 if len(feature_names) > 20 else len(feature_names)
        top_features = feature_importance_df.head(top_n)
        
        # สร้างกราฟ feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(np.arange(top_n), top_features['Importance'], align='center')
        plt.yticks(np.arange(top_n), top_features['Feature'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.tight_layout()
        plt.savefig(f'outputs/png/ml/{model_name}_feature_importance.png')
    
    # รวบรวม metrics และบันทึกลงไฟล์
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc,
        'AP': ap
    }
    
    return metrics

def save_model(model, model_name):
    """
    บันทึกโมเดลลงไฟล์
    """
    # สร้างโฟลเดอร์สำหรับเก็บโมเดล
    os.makedirs('data/models/ml', exist_ok=True)
    
    # บันทึกโมเดล
    model_path = f'data/models/ml/{model_name}.joblib'
    joblib.dump(model, model_path)
    print(f"บันทึกโมเดล {model_name} ไปยัง {model_path}")
    
    return model_path

def compare_models(all_metrics):
    """
    เปรียบเทียบประสิทธิภาพของโมเดลต่างๆ
    """
    print("\n===== เปรียบเทียบประสิทธิภาพของโมเดล =====")
    
    # สร้าง DataFrame จาก metrics
    metrics_df = pd.DataFrame(all_metrics)
    
    # บันทึกผลการเปรียบเทียบ
    metrics_df.to_csv('outputs/csv/ml/model_comparison.csv', index=False)
    
    # แสดงผลการเปรียบเทียบในรูปแบบตาราง
    print(metrics_df)
    
    # เตรียมข้อมูลสำหรับการสร้างกราฟ
    models = metrics_df['Model']
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'AP']
    
    # สร้างกราฟแท่งเปรียบเทียบ
    plt.figure(figsize=(14, 8))
    bar_width = 0.15
    index = np.arange(len(models))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(index + i*bar_width, metrics_df[metric], bar_width, label=metric)
    
    plt.xlabel('Models')  
    plt.ylabel('Score')   
    plt.title('Performance Comparison of Models')  
    plt.xticks(index + bar_width * (len(metrics_to_plot) - 1) / 2, models)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/png/ml/model_comparison.png')
    
    # สร้างกราฟเฉพาะ F1 Score
    plt.figure(figsize=(10, 6))
    plt.bar(models, metrics_df['F1'], color='skyblue')
    plt.xlabel('Models')  # Changed from 'โมเดล'
    plt.ylabel('F1 Score')
    plt.title('F1 Score of Models')  
    plt.ylim(0, 1.0)
    for i, v in enumerate(metrics_df['F1']):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    plt.tight_layout()
    plt.savefig('outputs/png/ml/f1_comparison.png')
    
    return metrics_df

def main():
    """
    ฟังก์ชันหลักสำหรับการฝึกและประเมินโมเดลทั้งหมด
    """
    print("===== เริ่มการฝึกโมเดลสำหรับการพยากรณ์โอกาสที่ฝนจะตก =====")
    
    # โหลดข้อมูล
    X_train, X_test, y_train, y_test, feature_names = load_processed_data()
    
    if X_train is None:
        return
    
    print(f"จำนวนข้อมูล: Train {X_train.shape[0]} ตัวอย่าง, Test {X_test.shape[0]} ตัวอย่าง")
    print(f"จำนวน features: {X_train.shape[1]}")
    
    # เก็บ metrics ของทุกโมเดล
    all_metrics = []
    
    # ฝึกและประเมิน Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "LogisticRegression", feature_names)
    save_model(lr_model, "LogisticRegression")
    all_metrics.append(lr_metrics)
    
    # ฝึกและประเมิน Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "RandomForest", feature_names)
    save_model(rf_model, "RandomForest")
    all_metrics.append(rf_metrics)
    
    # ฝึกและประเมิน XGBoost
    xgb_model = train_gradient_boosting(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost", feature_names)
    save_model(xgb_model, "XGBoost")
    all_metrics.append(xgb_metrics)
    
    # ฝึกและประเมิน Naive Bayes
    nb_model = train_naive_bayes(X_train, y_train)
    nb_metrics = evaluate_model(nb_model, X_test, y_test, "NaiveBayes", feature_names)
    save_model(nb_model, "NaiveBayes")
    all_metrics.append(nb_metrics)
    
    # เปรียบเทียบโมเดล
    compare_models(all_metrics)
    
    print("\n===== การฝึกและประเมินโมเดลเสร็จสิ้น =====")
    print("ผลลัพธ์ถูกบันทึกไว้ในโฟลเดอร์ outputs/")

if __name__ == "__main__":
    main()