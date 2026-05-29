"""
RoadWatch — FastAPI Backend
National Road Safety Hackathon 2026 | CoERS, IIT Madras
"""
import os, json, re, time, uuid, hashlib
try:
    from langdetect import detect as langdetect_detect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    def langdetect_detect(t): return 'en'
from difflib import get_close_matches
from datetime import datetime, date
from typing import Optional, List
from collections import defaultdict
from math import log1p

import numpy as np
import joblib
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(BASE, "templates", "index.html")
ML_DIR = os.path.join(BASE, "ml", "trained")
DATA_DIR = os.path.join(BASE, "data")
STATIC_DIR = os.path.join(BASE, "static")

# ── App ──
app = FastAPI(title="RoadWatch API", version="1.0.0", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── UTF-8 Enforcement Middleware ──
@app.middleware("http")
async def enforce_utf8(request: Request, call_next):
    response = await call_next(request)
    ct = response.headers.get("content-type", "")
    if "application/json" in ct and "charset" not in ct:
        response.headers["content-type"] = ct + "; charset=utf-8"
    return response

# ── Language Detection Helper ──
SUPPORTED_LANGS = {"en", "hi", "bn", "ta", "te", "mr"}

def detect_language(text: str, accept_lang: str = None, explicit_lang: str = None) -> str:
    """Detect input language. Priority: explicit > content detection > Accept-Language > default."""
    if explicit_lang and explicit_lang in SUPPORTED_LANGS:
        return explicit_lang
    # Try content-based detection
    if HAS_LANGDETECT and text and len(text.strip()) > 5:
        try:
            detected = langdetect_detect(text)
            lang_code = detected.split("-")[0]
            if lang_code in SUPPORTED_LANGS:
                return lang_code
        except Exception:
            pass
    # Try Accept-Language header
    if accept_lang:
        for part in accept_lang.split(","):
            code = part.strip().split(";")[0].split("-")[0].strip().lower()
            if code in SUPPORTED_LANGS:
                return code
    return "en"

# ── Rate Limiter ──
rate_store = defaultdict(list)
RATE_LIMIT = 30
RATE_WINDOW = 60

def check_rate(ip: str):
    now = time.time()
    rate_store[ip] = [t for t in rate_store[ip] if now - t < RATE_WINDOW]
    if len(rate_store[ip]) >= RATE_LIMIT:
        raise HTTPException(429, "Rate limit exceeded. Try again later.")
    rate_store[ip].append(now)

# ── Input Sanitizer ──
INJECT_RE = re.compile(r'[<>"\';&|`$\\]')
PHONE_RE = re.compile(r"^\+?[0-9\s\-]{10,15}$")

def sanitize(v: str) -> str:
    if INJECT_RE.search(v):
        raise ValueError("Invalid characters detected")
    return v.strip()

# ── Load ML Models ──
models = {}
def load_models():
    global models
    try:
        models = {
            "anomaly": joblib.load(os.path.join(ML_DIR, "anomaly_model.pkl")),
            "anomaly_scaler": joblib.load(os.path.join(ML_DIR, "anomaly_scaler.pkl")),
            "road_type": joblib.load(os.path.join(ML_DIR, "road_type_model.pkl")),
            "road_type_scaler": joblib.load(os.path.join(ML_DIR, "road_type_scaler.pkl")),
            "road_class_names": joblib.load(os.path.join(ML_DIR, "road_class_names.pkl")),
            "severity": joblib.load(os.path.join(ML_DIR, "severity_model.pkl")),
            "severity_scaler": joblib.load(os.path.join(ML_DIR, "severity_scaler.pkl")),
            "severity_names": joblib.load(os.path.join(ML_DIR, "severity_names.pkl")),
            "lifecycle": joblib.load(os.path.join(ML_DIR, "lifecycle_model.pkl")),
            "lifecycle_scaler": joblib.load(os.path.join(ML_DIR, "lifecycle_scaler.pkl")),
        }
        print(f"  Loaded {len(models)} model artifacts")
    except Exception as e:
        print(f"  Warning: Could not load models: {e}")

# ── Load Data ──
data_store = {}
def load_data():
    global data_store
    for fn in ["authorities.json","pmgsy_roads.json","contractor_index.json","rainfall.json","osm_roads.json"]:
        path = os.path.join(DATA_DIR, fn)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data_store[fn.replace(".json","")] = json.load(f)
    print(f"  Loaded {len(data_store)} data files")

# ── Complaints Store ──
complaints = []
complaint_counter = 0

# ── Pydantic Models ──
class ChatMsg(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    context: Optional[str] = None
    lang: Optional[str] = None
    @field_validator("message")
    @classmethod
    def clean(cls, v): return sanitize(v)

class AnomalyReq(BaseModel):
    spend_per_km_lakh: float = Field(..., ge=0.1, le=500)
    road_age_months: int = Field(..., ge=0, le=600)
    rainfall_mm: int = Field(..., ge=50, le=5000)
    traffic_class: int = Field(..., ge=0, le=2)
    road_type: int = Field(..., ge=0, le=4)
    maintenance_done: int = Field(..., ge=0, le=1)
    observed_quality: float = Field(..., ge=0, le=10)

class RoadTypeReq(BaseModel):
    osm_tag_enc: int = Field(..., ge=0, le=9)
    width_m: float = Field(..., ge=1, le=50)
    speed_limit: int = Field(..., ge=5, le=150)
    lanes: int = Field(..., ge=1, le=10)

class SeverityReq(BaseModel):
    issue_type_enc: int = Field(..., ge=0, le=6)
    area_sqm: float = Field(..., ge=0.1, le=200)
    report_count: int = Field(..., ge=1, le=500)
    days_since_repair: int = Field(..., ge=0, le=3650)
    near_school: int = Field(..., ge=0, le=1)
    near_hospital: int = Field(..., ge=0, le=1)

class LifecycleReq(BaseModel):
    material_enc: int = Field(..., ge=0, le=4)
    age_months: int = Field(..., ge=0, le=600)
    rainfall_mm: int = Field(..., ge=50, le=5000)
    spend_per_km: float = Field(..., ge=0.1, le=500)
    traffic_class: int = Field(..., ge=0, le=2)

class ComplaintReq(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    phone: str = Field(..., min_length=10, max_length=15)
    district: str = Field(..., min_length=2, max_length=50)
    road_name: str = Field(..., min_length=2, max_length=200)
    road_type: str = Field(..., pattern=r"^(NH|SH|MDR|ODR|Village)$")
    issue_type: str = Field(..., min_length=2, max_length=50)
    description: str = Field(..., min_length=10, max_length=2000)
    area_sqm: float = Field(1.0, ge=0.1, le=200)
    near_school: int = Field(0, ge=0, le=1)
    near_hospital: int = Field(0, ge=0, le=1)

    @field_validator("name","district","road_name","description","issue_type")
    @classmethod
    def clean_text(cls, v): return sanitize(v)

    @field_validator("phone")
    @classmethod
    def clean_phone(cls, v):
        if not PHONE_RE.match(v):
            raise ValueError("Invalid phone number format")
        return v

# ── Smart Chatbot (TF-IDF + Intent + Memory) ──
from ml.chatbot import smart_chat

# ── Authority lookup ──
AUTHORITY_MAP = {
    "NH": {"authority":"NHAI Regional Office, Chennai","officer":"Regional Officer, NHAI","helpline":"1033","portal":"nhai.gov.in"},
    "SH": {"authority":"TN Highways Dept, Chief Engineer Office","officer":"Chief Engineer (Highways)","helpline":"1800-425-0101","portal":"tnhighways.gov.in"},
    "MDR": {"authority":"District Executive Engineer, PWD","officer":"Executive Engineer, PWD Division","helpline":"1800-425-0101","portal":"tnpwd.gov.in"},
    "ODR": {"authority":"District Executive Engineer, PWD","officer":"Executive Engineer, PWD Division","helpline":"1800-425-0101","portal":"tnpwd.gov.in"},
    "Village": {"authority":"Gram Panchayat / DRDA","officer":"Block Development Officer","helpline":"104","portal":"tnrd.gov.in"},
}

INTERNATIONAL_AUTHORITY_MAP = {
    "National Highway / Motorway": {"authority": "National Transport Authority / Federal Highway Agency", "officer": "Highway Administrator", "helpline": "Emergency Services (112/911)", "portal": "Local Gov Transport Portal"},
    "State / Regional Highway": {"authority": "State/Provincial Transport Department", "officer": "Regional Engineer", "helpline": "Emergency Services (112/911)", "portal": "Local Gov Transport Portal"},
    "Major District Road": {"authority": "County / District Transport Office", "officer": "District Engineer", "helpline": "Local Police/Emergency", "portal": "Local Gov Transport Portal"},
    "Local / City Road": {"authority": "Municipal Corporation / City Council", "officer": "City Engineer", "helpline": "City Services (311/Non-Emergency)", "portal": "City Gov Portal"},
    "Rural / Village Road": {"authority": "Rural Council / Local Parish", "officer": "Local Administrator", "helpline": "Local Services", "portal": "Local Gov Portal"}
}

ISSUE_TYPES = {"pothole":0,"crack":1,"waterlogging":2,"broken_edge":3,"missing_signage":4,"faded_markings":5,"debris":6}
SEVERITY_SLA = {0:"30 days",1:"14 days",2:"7 days",3:"48 hours"}
ROAD_TYPES_LIST = ["NH","SH","MDR","ODR","Village"]
MATERIAL_NAMES = ["Bitumen","Concrete","Gravel","WBM","BM"]
TRAFFIC_NAMES = ["Low","Medium","High"]

# ── Startup ──
@app.on_event("startup")
def startup():
    print("\n[*] RoadWatch API Starting...")
    load_models()
    load_data()
    print("[OK] RoadWatch API Ready\n")

# ── Routes ──
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if os.path.exists(TEMPLATE):
        with open(TEMPLATE, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>RoadWatch — Template not found. Run the build first.</h1>")

@app.get("/api/health")
async def health():
    loaded = list(set(k.split("_")[0] for k in models.keys())) if models else []
    return {"status":"ok","models_loaded":len(models)>0,"model_names":loaded,"data_loaded":len(data_store),"timestamp":datetime.now().isoformat()}

@app.get("/api/metadata")
async def metadata():
    return {"road_types":ROAD_TYPES_LIST,"issue_types":list(ISSUE_TYPES.keys()),"materials":MATERIAL_NAMES,"traffic_classes":TRAFFIC_NAMES,"severity_levels":["Low","Medium","High","Critical"],"sla":SEVERITY_SLA,"authorities":AUTHORITY_MAP}

@app.post("/api/chat")
async def chat(req: ChatMsg, request: Request):
    check_rate(request.client.host)
    ctx = None
    if req.context:
        try:
            ctx = json.loads(req.context)
        except (json.JSONDecodeError, TypeError):
            ctx = None
    session = hashlib.md5(request.client.host.encode()).hexdigest()[:8]
    accept_lang = request.headers.get("accept-language", "")
    detected_lang = detect_language(req.message, accept_lang=accept_lang, explicit_lang=req.lang)
    response_text = smart_chat(req.message, context=ctx, session_id=session, authority_map=AUTHORITY_MAP, lang=detected_lang)
    return {"response": response_text, "detected_language": detected_lang, "timestamp": datetime.now().isoformat()}

@app.post("/api/predict/anomaly")
async def predict_anomaly(req: AnomalyReq, request: Request):
    check_rate(request.client.host)
    if not models:
        raise HTTPException(503, "Models not loaded")
    X = np.array([[req.spend_per_km_lakh, req.road_age_months, req.rainfall_mm, req.traffic_class, req.road_type, req.maintenance_done, req.observed_quality]])
    X_s = models["anomaly_scaler"].transform(X)
    pred = int(models["anomaly"].predict(X_s)[0])
    proba = float(models["anomaly"].predict_proba(X_s)[0][1]) * 100
    exp_q = max(0.1, min(10, 10 - (req.road_age_months/12)*0.6 - (req.rainfall_mm/1000)*0.8 + log1p(req.spend_per_km_lakh)*0.9 + req.maintenance_done*1.5 - req.traffic_class*0.4))
    gap = exp_q - req.observed_quality
    if gap > 2.5 and pred == 1:
        verdict = "Suspected Budget Leakage"
        rec = "File RTI for contractor details and expenditure breakdown."
        color = "red"
    elif pred == 1:
        verdict = "Quality Below Expected"
        rec = "Monitor and re-report in 30 days."
        color = "orange"
    else:
        verdict = "Road Quality Within Normal Range"
        rec = "No action required."
        color = "green"
    return {"anomaly_flag":pred,"anomaly_probability":round(proba,1),"expected_quality":round(exp_q,1),"observed_quality":req.observed_quality,"quality_gap":round(gap,1),"verdict":verdict,"recommendation":rec,"color":color,"road_type":ROAD_TYPES_LIST[req.road_type],"traffic":TRAFFIC_NAMES[req.traffic_class]}

@app.post("/api/predict/road-type")
async def predict_road_type(req: RoadTypeReq, request: Request):
    check_rate(request.client.host)
    if not models:
        raise HTTPException(503, "Models not loaded")
    X = np.array([[req.osm_tag_enc, req.width_m, req.speed_limit, req.lanes]])
    X_s = models["road_type_scaler"].transform(X)
    pred = int(models["road_type"].predict(X_s)[0])
    proba = models["road_type"].predict_proba(X_s)[0]
    name = models["road_class_names"][pred]
    auth = AUTHORITY_MAP.get(name, {})
    return {"road_class":pred,"road_type":name,"confidence":round(float(max(proba))*100,1),"all_probabilities":{models["road_class_names"][i]:round(float(p)*100,1) for i,p in enumerate(proba)},"authority":auth}

@app.post("/api/predict/severity")
async def predict_severity(req: SeverityReq, request: Request):
    check_rate(request.client.host)
    if not models:
        raise HTTPException(503, "Models not loaded")
    X = np.array([[req.issue_type_enc, req.area_sqm, req.report_count, req.days_since_repair, req.near_school, req.near_hospital]])
    X_s = models["severity_scaler"].transform(X)
    pred = int(models["severity"].predict(X_s)[0])
    proba = models["severity"].predict_proba(X_s)[0]
    name = models["severity_names"][pred]
    sla = SEVERITY_SLA[pred]
    issue_names = list(ISSUE_TYPES.keys())
    return {"severity":pred,"severity_label":name,"sla":sla,"confidence":round(float(max(proba))*100,1),"issue_type":issue_names[req.issue_type_enc] if req.issue_type_enc < len(issue_names) else "unknown","all_probabilities":{models["severity_names"][i]:round(float(p)*100,1) for i,p in enumerate(proba)}}

@app.post("/api/predict/lifecycle")
async def predict_lifecycle(req: LifecycleReq, request: Request):
    check_rate(request.client.host)
    if not models:
        raise HTTPException(503, "Models not loaded")
    X = np.array([[req.material_enc, req.age_months, req.rainfall_mm, req.spend_per_km, req.traffic_class]])
    X_s = models["lifecycle_scaler"].transform(X)
    pred = float(models["lifecycle"].predict(X_s)[0])
    pred = max(0, pred)
    years = pred / 12
    if pred < 6:
        urgency = "CRITICAL — Immediate repair needed"
        color = "red"
    elif pred < 24:
        urgency = "HIGH — Schedule repair within 6 months"
        color = "orange"
    elif pred < 60:
        urgency = "MEDIUM — Plan maintenance within 2 years"
        color = "yellow"
    else:
        urgency = "LOW — Road in good condition"
        color = "green"
    return {"months_to_failure":round(pred,1),"years_to_failure":round(years,1),"urgency":urgency,"color":color,"material":MATERIAL_NAMES[req.material_enc],"traffic":TRAFFIC_NAMES[req.traffic_class]}

@app.get("/api/roads/pmgsy")
async def get_pmgsy(district: Optional[str] = None):
    d = data_store.get("pmgsy_roads", {})
    roads = d.get("roads", [])
    if district:
        filtered = [r for r in roads if r.get("district","").lower() == district.lower()] or roads
    else:
        filtered = roads
    return {"source":d.get("source","OMMAS — pmgsy.dord.gov.in"),"retrieved":d.get("retrieved",""),"district":district or "Pan-India","total":len(filtered),"roads":filtered}

@app.get("/api/roads/contractors")
async def get_contractors():
    d = data_store.get("contractor_index", {})
    return {"source":d.get("source",""),"retrieved":d.get("retrieved",""),"grading_criteria":d.get("grading_criteria",{}),"contractors":d.get("contractors",[])}

@app.get("/api/roads/authorities")
async def get_authorities(road_type: Optional[str] = None):
    d = data_store.get("authorities", {})
    auths = d.get("authorities", {})
    if road_type and road_type in auths:
        return {"authority": auths[road_type], "districts": d.get("districts", []), "emergency": d.get("emergency_contacts", {})}
    return {"authorities": auths, "districts": d.get("districts", []), "emergency": d.get("emergency_contacts", {})}

@app.get("/api/roads/rainfall")
async def get_rainfall(district: Optional[str] = None):
    d = data_store.get("rainfall", {})
    districts = d.get("districts", [])
    if district:
        filtered = [x for x in districts if x["name"].lower() == district.lower()]
        return {"source":d.get("source",""),"retrieved":d.get("retrieved",""),"districts":filtered or districts}
    return {"source":d.get("source",""),"retrieved":d.get("retrieved",""),"total":len(districts),"districts":districts}

@app.post("/api/complaints/submit")
async def submit_complaint(req: ComplaintReq, request: Request):
    check_rate(request.client.host)
    global complaint_counter
    complaint_counter += 1
    cid = f"RW{datetime.now().strftime('%Y%m%d')}{complaint_counter:04d}"
    masked_phone = "****" + req.phone[-4:]
    auth = AUTHORITY_MAP.get(req.road_type, {})
    # Run severity model
    sev_label = "Medium"
    sla = "14 days"
    sev_val = 1
    if models:
        it_enc = ISSUE_TYPES.get(req.issue_type.lower().replace(" ","_"), 0)
        X = np.array([[it_enc, req.area_sqm, 1, 180, req.near_school, req.near_hospital]])
        X_s = models["severity_scaler"].transform(X)
        sev_val = int(models["severity"].predict(X_s)[0])
        sev_label = models["severity_names"][sev_val]
        sla = SEVERITY_SLA[sev_val]
    complaint = {"id":cid,"name":req.name,"phone":masked_phone,"district":req.district,"road_name":req.road_name,"road_type":req.road_type,"issue_type":req.issue_type,"description":req.description,"severity":sev_label,"severity_level":sev_val,"routed_to":auth.get("authority","Unknown"),"officer":auth.get("officer",""),"helpline":auth.get("helpline",""),"portal":auth.get("portal",""),"sla":sla,"status":"Filed","filed_at":datetime.now().isoformat(),"area_sqm":req.area_sqm}
    complaints.append(complaint)
    return {"complaint_id":cid,"severity":sev_label,"routed_to":auth.get("authority",""),"officer":auth.get("officer",""),"helpline":auth.get("helpline",""),"expected_resolution":sla,"message":f"Complaint {cid} filed successfully. Routed to {auth.get('officer','')}. Expected resolution: {sla}."}

@app.get("/api/complaints/list")
async def list_complaints():
    return {"total":len(complaints),"complaints":complaints}

@app.get("/api/complaints/{cid}")
async def get_complaint(cid: str):
    for c in complaints:
        if c["id"] == cid:
            return c
    raise HTTPException(404, "Complaint not found")

@app.get("/api/dashboard/summary")
async def dashboard_summary():
    pmgsy = data_store.get("pmgsy_roads", {})
    roads = pmgsy.get("roads", [])
    contractors = data_store.get("contractor_index", {}).get("contractors", [])
    total_sanctioned = sum(r.get("sanctioned_cost_lakh", 0) for r in roads)
    total_spent = sum(r.get("expenditure_lakh", 0) for r in roads)
    avg_quality = np.mean([r.get("quality_score", 5) for r in roads]) if roads else 0
    flagged = sum(1 for c in contractors if c.get("flag", False))
    return {
        "total_roads_monitored": len(roads),
        "total_sanctioned_lakh": round(total_sanctioned, 2),
        "total_spent_lakh": round(total_spent, 2),
        "avg_quality_score": round(float(avg_quality), 1),
        "complaints_filed": len(complaints),
        "contractors_flagged": flagged,
        "total_contractors": len(contractors),
        "district": "Chengalpattu",
        "state": "Tamil Nadu",
        "source": "OMMAS — pmgsy.dord.gov.in",
        "retrieved": pmgsy.get("retrieved", datetime.now().strftime("%Y-%m-%d")),
        "models_active": len(models) > 0,
        "uk_data_available": True
    }

@app.get("/api/geo/reverse")
async def reverse_geocode(lat: float, lon: float):
    """Reverse geocode coordinates using Nominatim (OSM) - free, no API key."""
    import requests as req
    check_rate("geo")
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&addressdetails=1&zoom=18"
        resp = req.get(url, headers={"User-Agent": "RoadWatch/1.0 (hackathon)"}, timeout=10)
        data = resp.json()
        addr = data.get("address", {})
        road = addr.get("road", addr.get("highway", addr.get("pedestrian", "Unknown Road")))
        district = addr.get("county", addr.get("state_district", addr.get("city", addr.get("town", "Unknown"))))
        state = addr.get("state", "Unknown")
        suburb = addr.get("suburb", addr.get("neighbourhood", ""))
        postcode = addr.get("postcode", "")
        country_code = addr.get("country_code", "in")
        osm_type = data.get("type", "")
        
        road_type = "Village"
        if country_code == "in":
            # Indian classification
            rl = road.lower()
            if any(x in rl for x in ["nh ", "nh-", "national highway"]):
                road_type = "NH"
            elif any(x in rl for x in ["sh ", "sh-", "state highway"]):
                road_type = "SH"
            elif any(x in rl for x in ["mdr", "major district"]):
                road_type = "MDR"
            elif any(x in rl for x in ["main road", "trunk road", "bypass", "expressway"]):
                road_type = "SH"
            elif any(x in rl for x in ["avenue", "street", "lane", "nagar", "salai"]):
                road_type = "ODR"
            auth = AUTHORITY_MAP.get(road_type, {})
        else:
            # International classification based on OSM type
            if osm_type in ["motorway", "trunk", "motorway_link", "trunk_link"]:
                road_type = "National Highway / Motorway"
            elif osm_type in ["primary", "secondary", "primary_link", "secondary_link"]:
                road_type = "State / Regional Highway"
            elif osm_type in ["tertiary", "unclassified", "tertiary_link"]:
                road_type = "Major District Road"
            elif osm_type in ["residential", "living_street", "pedestrian"]:
                road_type = "Local / City Road"
            elif osm_type in ["track", "path", "service"]:
                road_type = "Rural / Village Road"
            else:
                road_type = "Local / City Road"
            auth = INTERNATIONAL_AUTHORITY_MAP.get(road_type, {})

        # Fetch contractor and budget data for demonstration
        contractor = "Unknown Contractor"
        last_repaired = "Unknown"
        sanctioned = "N/A"
        spent = "N/A"
        
        if "pmgsy_roads" in data_store:
            roads = data_store["pmgsy_roads"].get("roads", [])
            road_names = [r.get("road_name", "") for r in roads if r.get("road_name")]
            if road_names:
                matches = get_close_matches(road, road_names, n=1, cutoff=0.4)
                if matches:
                    matched = next((r for r in roads if r.get("road_name") == matches[0]), None)
                    if matched:
                        contractor = matched.get("contractor", "L&T Infrastructure (Demo)")
                        last_repaired = matched.get("last_repaired", "2024-02-15")
                        sanctioned = f"₹{matched.get('sanctioned_cost_lakh', 0)}L"
                        spent = f"₹{matched.get('expenditure_lakh', 0)}L"
                else:
                    # Fallback to realistic demo data if no exact match
                    contractor = "L&T Infrastructure (Demo)"
                    last_repaired = "2023-11-20"
                    sanctioned = "₹120L"
                    spent = "₹108L"
        if country_code != "in":
             contractor = "International Construction Co."
             last_repaired = "2023-08-10"
             sanctioned = "$1.5M"
             spent = "$1.2M"

        return {
            "lat": lat, "lon": lon, "road": road, "road_type": road_type,
            "district": district, "state": state, "suburb": suburb,
            "postcode": postcode, "full_address": data.get("display_name", ""),
            "authority": auth, "osm_type": osm_type,
            "category": data.get("category", ""),
            "contractor": contractor,
            "last_repaired": last_repaired,
            "sanctioned": sanctioned,
            "spent": spent
        }
    except Exception as e:
        return {"error": str(e), "lat": lat, "lon": lon}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
