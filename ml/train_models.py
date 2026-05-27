"""
RoadWatch ML Training Pipeline
Trains 4 scikit-learn models on synthetic data based on PMGSY statistics.
Usage: python ml/train_models.py
"""
import os, json, numpy as np, pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_DIR = os.path.join(BASE_DIR, "trained")
os.makedirs(TRAINED_DIR, exist_ok=True)
np.random.seed(42)

def train_anomaly_model():
    print("\n" + "="*60 + "\nTraining Model 1: Budget Anomaly Detector\n" + "="*60)
    n = 3000
    spend = np.random.uniform(0.1, 500, n)
    age = np.random.randint(0, 601, n)
    rain = np.random.randint(50, 5001, n)
    traffic = np.random.randint(0, 3, n)
    rtype = np.random.randint(0, 5, n)
    maint = np.random.randint(0, 2, n)
    exp_q = np.clip(10 - (age/12)*0.6 - (rain/1000)*0.8 + np.log1p(spend)*0.9 + maint*1.5 - traffic*0.4, 0, 10)
    obs_q = np.clip(exp_q + np.random.normal(0, 0.8, n), 0, 10)
    mask = np.random.choice([True, False], size=n, p=[0.2, 0.8])
    obs_q[mask] = np.clip(obs_q[mask] * np.random.uniform(0.15, 0.55, mask.sum()), 0, 10)
    safe_exp = np.where(exp_q > 0.1, exp_q, 0.1)
    labels = ((safe_exp - obs_q) / safe_exp * 100 > 40).astype(int)
    X = np.column_stack([spend, age, rain, traffic, rtype, maint, obs_q])
    feats = ["spend_per_km_lakh","road_age_months","rainfall_mm","traffic_class","road_type","maintenance_done","observed_quality"]
    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    m = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42, class_weight="balanced")
    m.fit(Xtr_s, ytr)
    acc = accuracy_score(yte, m.predict(Xte_s))
    print(f"  Accuracy: {acc:.4f}")
    joblib.dump(m, os.path.join(TRAINED_DIR, "anomaly_model.pkl"))
    joblib.dump(sc, os.path.join(TRAINED_DIR, "anomaly_scaler.pkl"))
    return acc, feats

def train_road_type_model():
    print("\n" + "="*60 + "\nTraining Model 2: Road Type Classifier\n" + "="*60)
    n = 2000
    rc = {0:{"w":(7,45),"s":(60,120),"l":(2,8),"t":[0,1]},1:{"w":(5,20),"s":(40,100),"l":(2,6),"t":[2,3]},2:{"w":(3.5,12),"s":(30,80),"l":(1,4),"t":[4,5]},3:{"w":(3,7),"s":(20,60),"l":(1,2),"t":[6,7]},4:{"w":(2,5),"s":(10,40),"l":(1,2),"t":[8,9]}}
    data = []
    for cid, p in rc.items():
        for _ in range(n//5):
            data.append([np.random.choice(p["t"]), np.random.uniform(*p["w"]), np.random.randint(*p["s"]), np.random.randint(*p["l"]), cid])
    df = pd.DataFrame(data, columns=["osm_tag_enc","width_m","speed_limit","lanes","road_class"])
    X = df[["osm_tag_enc","width_m","speed_limit","lanes"]].values; y = df["road_class"].values
    cn = ["NH","SH","MDR","ODR","Village"]
    feats = ["osm_tag_enc","width_m","speed_limit","lanes"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    m = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=4, random_state=42)
    m.fit(Xtr_s, ytr)
    acc = accuracy_score(yte, m.predict(Xte_s))
    print(f"  Accuracy: {acc:.4f}")
    joblib.dump(m, os.path.join(TRAINED_DIR, "road_type_model.pkl"))
    joblib.dump(sc, os.path.join(TRAINED_DIR, "road_type_scaler.pkl"))
    joblib.dump(cn, os.path.join(TRAINED_DIR, "road_class_names.pkl"))
    return acc, feats

def train_severity_model():
    print("\n" + "="*60 + "\nTraining Model 3: Complaint Severity Classifier\n" + "="*60)
    n = 2500
    it = np.random.randint(0, 7, n)
    area = np.clip(np.random.exponential(5, n), 0.1, 200)
    rc = np.random.randint(1, 100, n)
    days = np.random.randint(0, 3650, n)
    ns = np.random.randint(0, 2, n)
    nh = np.random.randint(0, 2, n)
    ss = area*0.15 + rc*0.08 + days/365*0.5 + ns*2.0 + nh*2.5 + np.where(it==0,2.0,0) + np.where(it==2,1.5,0) + np.where(it==4,1.8,0) + np.random.normal(0,1.0,n)
    labels = np.zeros(n, dtype=int)
    p25, p50, p75 = np.percentile(ss, [25, 50, 75])
    labels[ss >= p75] = 3; labels[(ss >= p50) & (ss < p75)] = 2; labels[(ss >= p25) & (ss < p50)] = 1
    sn = ["Low","Medium","High","Critical"]
    feats = ["issue_type_enc","area_sqm","report_count","days_since_repair","near_school","near_hospital"]
    X = np.column_stack([it, area, rc, days, ns, nh])
    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    m = RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_split=4, min_samples_leaf=2, random_state=42, class_weight="balanced")
    m.fit(Xtr_s, ytr)
    acc = accuracy_score(yte, m.predict(Xte_s))
    print(f"  Accuracy: {acc:.4f}")
    joblib.dump(m, os.path.join(TRAINED_DIR, "severity_model.pkl"))
    joblib.dump(sc, os.path.join(TRAINED_DIR, "severity_scaler.pkl"))
    joblib.dump(sn, os.path.join(TRAINED_DIR, "severity_names.pkl"))
    return acc, feats

def train_lifecycle_model():
    print("\n" + "="*60 + "\nTraining Model 4: Road Lifecycle Predictor\n" + "="*60)
    n = 2000
    ml = {0:84, 1:120, 2:36, 3:48, 4:60}
    mat = np.random.randint(0, 5, n)
    age = np.random.randint(0, 601, n)
    rain = np.random.randint(50, 5001, n)
    spend = np.random.uniform(0.1, 500, n)
    traffic = np.random.randint(0, 3, n)
    base = np.array([ml[m] for m in mat], dtype=float)
    y = np.clip(base - age*0.4 - (rain/1000)*8 + np.log1p(spend)*3 - traffic*6 + np.random.normal(0,5,n), 0, 300)
    feats = ["material_enc","age_months","rainfall_mm","spend_per_km","traffic_class"]
    X = np.column_stack([mat, age, rain, spend, traffic])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    m = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42)
    m.fit(Xtr_s, ytr)
    mae = mean_absolute_error(yte, m.predict(Xte_s))
    print(f"  MAE: {mae:.2f} months")
    joblib.dump(m, os.path.join(TRAINED_DIR, "lifecycle_model.pkl"))
    joblib.dump(sc, os.path.join(TRAINED_DIR, "lifecycle_scaler.pkl"))
    return mae, feats

if __name__ == "__main__":
    print("="*60 + "\n  RoadWatch ML Training Pipeline v1.0\n  National Road Safety Hackathon 2026 | CoERS, IITM\n" + "="*60)
    a1, f1 = train_anomaly_model()
    a2, f2 = train_road_type_model()
    a3, f3 = train_severity_model()
    a4, f4 = train_lifecycle_model()
    meta = {"trained_at": datetime.now().isoformat(), "framework": "scikit-learn 1.5.2",
        "models": {
            "anomaly_detector": {"type":"RandomForestClassifier","accuracy":round(a1,4),"features":f1,"training_samples":3000,"files":["anomaly_model.pkl","anomaly_scaler.pkl"]},
            "road_type_classifier": {"type":"RandomForestClassifier","accuracy":round(a2,4),"features":f2,"training_samples":2000,"files":["road_type_model.pkl","road_type_scaler.pkl","road_class_names.pkl"]},
            "severity_classifier": {"type":"RandomForestClassifier","accuracy":round(a3,4),"features":f3,"training_samples":2500,"files":["severity_model.pkl","severity_scaler.pkl","severity_names.pkl"]},
            "lifecycle_predictor": {"type":"RandomForestRegressor","mae_months":round(a4,2),"features":f4,"training_samples":2000,"files":["lifecycle_model.pkl","lifecycle_scaler.pkl"]}
        }}
    with open(os.path.join(TRAINED_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\n" + "="*60 + "\nALL MODELS TRAINED SUCCESSFULLY\n" + "="*60)
    print(f"  Anomaly: {a1:.4f} | Road Type: {a2:.4f} | Severity: {a3:.4f} | Lifecycle MAE: {a4:.2f}mo")
    print(f"  Saved to: {TRAINED_DIR}")
