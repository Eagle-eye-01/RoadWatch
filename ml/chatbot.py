"""
RoadWatch Smart Chatbot Engine
TF-IDF Semantic Search + Intent Detection + Conversation Memory
Zero external API credits - runs entirely on scikit-learn locally.
"""
import os, json, re, numpy as np
from collections import defaultdict
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── LLM Integration (AI Context Injection) ──
def generate_llm_system_prompt(lang: str) -> str:
    """
    Constructs the system prompt wrapper that forces the underlying LLM to not only
    translate but process the semantic intent in the native language, preserving
    cultural context and colloquialisms before returning the generated payload.
    """
    lang_name = lang if lang and lang != 'en' else 'English'
    return f"""You are the RoadWatch AI Assistant, an expert in Indian road safety, infrastructure, and government policies.

Your primary objective is to process the user's semantic intent in their native language ({lang_name}), preserving cultural context, colloquialisms, and regional nuances, rather than performing a direct literal translation.

Directives:
1. Understand the user's intent in {lang_name}. If they use Romanized script (e.g., Hinglish, Tanglish), parse the underlying meaning.
2. Formulate your response focusing on empathy, clarity, and actionable advice related to road complaints, budget transparency, and contractor accountability.
3. Use culturally appropriate tone and terminology in {lang_name}.
4. Return the generated payload entirely in {lang_name} unless English technical terms are strictly necessary (e.g., PMGSY, OMMAS).
5. Always address the core issue before providing supplementary information.

Knowledge Context available:
- Road authorities (NHAI, State PWD, Gram Panchayat)
- Complaint routing and severity SLAs (Critical: 48 hours, Low: 30 days)
- Budget anomalies (comparing sanctioned cost vs observed quality)
- PMGSY 5-year maintenance rules and Contractor Accountability Index
"""

def call_llm_api(system_prompt: str, user_message: str, context: dict, authority_map: dict) -> str:
    """
    Mock function representing the actual LLM API call (e.g., OpenAI or Gemini).
    """
    import requests
    # In production, replace with actual SDK call (e.g., openai.ChatCompletion)
    # Using a hypothetical endpoint for demonstration of payload integrity
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        return None # Fallback to local NLP

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {json.dumps(context)}\nAuthorities: {json.dumps(authority_map)}\n\nQuery: {user_message}"}
        ],
        "temperature": 0.3
    }
    
    try:
        # Example of strictly enforcing UTF-8 payload integrity
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json; charset=utf-8"},
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            timeout=10
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM API Error: {e}")
    return None

# ── Knowledge Base ──
KNOWLEDGE = [
    {"keys":["authority","who is responsible","complaint to whom","report to","nhai","pwd","panchayat","responsible","which department","government body"],
     "topic":"road_authority",
     "answer":"Road authorities in India:\n\n**NH (National Highway)** - NHAI Regional Office (Helpline: 1033, Portal: nhai.gov.in)\n**SH (State Highway)** - TN Highways Dept, Chief Engineer (Helpline: 1800-425-0101)\n**MDR/ODR (District Roads)** - District Executive Engineer, PWD (Helpline: 1800-425-0101)\n**Village Roads** - Gram Panchayat / DRDA (Helpline: 104, Portal: tnrd.gov.in)\n\nYou can use the Road Type Classifier tool to automatically identify which authority handles a specific road."},
    {"keys":["pothole","crack","report","damage","broken","waterlog","defect","issue","road problem","bad road","poor condition","dangerous"],
     "topic":"report_defect",
     "answer":"Here's how to report a road defect:\n\n1. Navigate to the **File Complaint** section\n2. Enter the road details and describe the issue\n3. Our AI system automatically classifies the severity level\n4. The complaint gets routed to the responsible authority\n5. You'll receive a complaint ID and expected resolution timeline\n\nFor emergencies: Police 100 | Ambulance 108 | NHAI 1033"},
    {"keys":["budget","money","spent","cost","sanctioned","expenditure","lakh","crore","fund","spending","allocation","financial","corrupt","misuse","leakage"],
     "topic":"budget",
     "answer":"Road budget transparency data is sourced from:\n\n**OMMAS Portal** (pmgsy.dord.gov.in) - Shows sanctioned vs actual spent amounts for PMGSY roads\n**data.gov.in** - MoRTH road expenditure statistics at national level\n**GeoSadak** (geosadak-pmgsy.nic.in) - GPS-mapped road data with financial overlays\n\nOur **Budget Anomaly Detector** uses ML to compare spending against road quality. If there's a significant gap, it flags potential fund misuse and recommends filing an RTI."},
    {"keys":["pmgsy","5 year","five year","maintenance","warranty","contractor liable","escrow","pradhan mantri","gram sadak","rural road"],
     "topic":"pmgsy",
     "answer":"Under the Pradhan Mantri Gram Sadak Yojana (PMGSY):\n\nEvery contractor has a **mandatory 5-year maintenance contract** after road construction. The maintenance funds are held in a **separate escrow account** to prevent misuse. If a road deteriorates within 5 years, the **original contractor is legally liable** for repairs at their own cost.\n\nCitizens can file RTI requests to obtain contractor details, maintenance records, and inspection reports. Our Contractor Accountability Index tracks contractor performance across projects."},
    {"keys":["contractor","grade","blacklist","quality","accountability","index","rating","performance","builder","construction company"],
     "topic":"contractor",
     "answer":"Our Contractor Accountability Index grades contractors based on their track record:\n\n**Grade A** - Average quality score >= 8.0 (Excellent, preferred for new contracts)\n**Grade B** - Average quality >= 6.0 (Good, monitor periodically)\n**Grade C** - Average quality >= 4.0 (Below average, needs close monitoring)\n**Grade D** - Average quality < 4.0 (Poor, recommended for blacklisting)\n\nYou can view the complete contractor index with road-wise breakdowns in the Road Data section."},
    {"keys":["tambaram","chengalpattu","mudichur","kelambakkam","mamallapuram","omr","gst","chennai","kanchipuram","sholinganallur"],
     "topic":"local_roads",
     "answer":"For roads in the Tambaram/Chengalpattu/Chennai area:\n\n**GST Road (NH 45)** - NHAI Regional Office, Chennai (Helpline: 1033)\n**OMR / IT Expressway** - NHAI (Helpline: 1033)\n**ECR (East Coast Road)** - TN Highways Department\n**District roads (MDR/ODR)** - Executive Engineer, PWD Division, Chengalpattu\n**Village roads** - Local Gram Panchayat / DRDA\n\nYou can file a complaint on our platform and we'll automatically route it to the correct authority based on the road type."},
    {"keys":["emergency","helpline","phone","contact","call","number","accident","ambulance","police","fire"],
     "topic":"emergency",
     "answer":"Emergency Contacts:\n\nPolice: **100**\nAmbulance: **108**\nNHAI Helpline: **1033**\nTN PWD: **1800-425-0101**\nMoRTH (Ministry): **1800-11-6060**\nFire: **101**\nDisaster Management: **1078**\nWomen Helpline: **181**\n\nFor road accidents, call 108 first for medical help, then 100 for police. If the accident is on a National Highway, also call NHAI at 1033."},
    {"keys":["rti","right to information","transparency","file rti","information act","public information"],
     "topic":"rti",
     "answer":"How to file an RTI for road-related data:\n\n1. Visit **rtionline.gov.in**\n2. Select the relevant department (MoRTH for NH, State PWD for SH/MDR)\n3. Ask for: Contractor details, sanctioned amount, expenditure breakdown, quality inspection reports, maintenance records\n4. Pay the fee: Rs 10 (online payment)\n5. You should receive a response within 30 days\n\nOur Anomaly Detector can help identify which specific roads need RTI investigation by flagging spending-quality mismatches."},
    {"keys":["hello","hi","hey","help","what can you do","start","namaste","good morning","good evening","greetings"],
     "topic":"greeting",
     "answer":"Welcome to **RoadWatch AI**! I'm your road safety assistant. Here's what I can help you with:\n\n1. **Find road authorities** - Who's responsible for maintaining a specific road?\n2. **Budget transparency** - Track and verify government road spending\n3. **Report defects** - File complaints with automatic severity classification\n4. **Contractor accountability** - Check contractor grades and performance\n5. **Emergency contacts** - Quick access to all relevant helplines\n6. **PMGSY rules** - 5-year maintenance obligations and contractor liability\n7. **RTI guidance** - How to file RTI for road data\n\nJust ask me anything about roads, and I'll do my best to help! I also understand your GPS location if you've enabled it."},
    {"keys":["roadwatch","about","what is","project","hackathon","iit","how does it work","features","technology"],
     "topic":"about",
     "answer":"**RoadWatch** is an AI-powered road quality monitoring and budget transparency platform built for the National Road Safety Hackathon 2026 (CoERS, IIT Madras).\n\n**Technology Stack:**\n- 4 ML models (scikit-learn) for anomaly detection, road classification, severity prediction, and lifecycle forecasting\n- TF-IDF semantic search chatbot with fuzzy matching and GPS awareness\n- Budget-condition cross-referencing using PMGSY open data\n- Contractor accountability scoring system\n- Leaflet.js maps with satellite imagery and reverse geocoding\n\n**Data Sources:** OMMAS, data.gov.in, OpenStreetMap - all open government data, zero paid APIs."},
    {"keys":["how to use","guide","tutorial","steps","instructions","navigate","manual"],
     "topic":"guide",
     "answer":"Here's a quick guide to using RoadWatch:\n\n**Budget Anomaly Tab** - Enter road spending and quality data to detect potential fund misuse\n**Severity Tab** - Classify how urgent a road issue is (Low/Medium/High/Critical)\n**Lifecycle Tab** - Predict how many months a road has before it needs repair\n**File Complaint** - Submit a road complaint that gets auto-routed to the right authority\n**Road Data** - View PMGSY road data and contractor accountability index\n**Map & GPS** - Interactive map with satellite view and road identification\n**AI Assistant** - That's me! Ask me anything about roads.\n\nFor the best experience, enable GPS so I can provide location-specific information."},
    {"keys":["severity","critical","urgent","priority","sla","timeline","resolution time","how long"],
     "topic":"severity",
     "answer":"Our AI classifies road complaint severity into 4 levels:\n\n**Critical** - SLA: 48 hours (e.g., large potholes near schools/hospitals, road collapse)\n**High** - SLA: 7 days (e.g., significant waterlogging, missing signage on busy roads)\n**Medium** - SLA: 14 days (e.g., moderate cracks, faded markings)\n**Low** - SLA: 30 days (e.g., minor surface damage, small debris)\n\nFactors that increase severity: proximity to schools/hospitals, affected area size, number of citizen reports, and time since last repair."},
    {"keys":["lifecycle","lifespan","how long will road last","durability","prediction","failure","when will road break"],
     "topic":"lifecycle",
     "answer":"Our Road Lifecycle Predictor estimates remaining useful life based on:\n\n**Surface Material** - Concrete lasts longest (~10 years), followed by Bitumen (~7 years), then Gravel (~3 years)\n**Traffic Load** - Heavy traffic reduces lifespan significantly\n**Rainfall** - High rainfall areas see faster degradation\n**Construction Quality** - Measured by spend per km and contractor grade\n**Age** - Older roads naturally degrade faster\n\nThe model outputs months until the road likely needs repair, with urgency levels from LOW to CRITICAL."},
]

# ── Romanized Indic Keyword Mappings ──
# Maps Romanized Hindi/Tamil/Bengali/Telugu/Marathi road-related terms to English equivalents
INDIC_KEYWORDS = {
    # Hindi
    "sadak": "road", "sarak": "road",
    "garha": "pothole", "gadha": "pothole",
    "shikayat": "complaint",
    "paisa": "money", "budget": "budget",
    "sarkari": "government",
    "marammat": "repair",
    "daraar": "crack",
    "paani": "waterlogging",
    "khatarnak": "dangerous",
    "madad": "help",
    "suraksha": "safety",
    "thekedar": "contractor",
    # Tamil
    "saalai": "road",
    "palli": "pothole",
    "pugar": "complaint",
    "panam": "money",
    "sarkar": "government",
    "pazhutu": "defect",
    "thanni": "water",
    "aabathu": "danger",
    "uthavi": "help",
    # Bengali
    "rasta": "road",
    "gartha": "pothole",
    "obhijog": "complaint",
    "taka": "money",
    "meramot": "repair",
    "jol": "water",
    "bipod": "danger",
    "sahajjo": "help",
    # Telugu
    "rodu": "road",
    "gundhi": "pothole",
    "phiryadu": "complaint",
    "dabbu": "money",
    "pramadam": "danger",
    "sahayam": "help",
    # Marathi
    "khadda": "pothole",
    "takrar": "complaint",
    "durusti": "repair",
    "pani": "water",
    "dhoka": "danger",
}

# ── Native Script Keyword Detection ──
# Maps native Indic-script words directly to knowledge-base topics
NATIVE_SCRIPT_KEYWORDS = {
    # Hindi (Devanagari)
    "\u0938\u0921\u093c\u0915": "road_authority",       # सड़क
    "\u0917\u0921\u094d\u0922\u093e": "report_defect",   # गड्ढा
    "\u0936\u093f\u0915\u093e\u092f\u0924": "report_defect", # शिकायत
    "\u092c\u091c\u091f": "budget",                     # बजट
    "\u0920\u0947\u0915\u0947\u0926\u093e\u0930": "contractor", # ठेकेदार
    "\u092e\u0926\u0926": "greeting",                   # मदद
    "\u0939\u0947\u0932\u094d\u092a\u0932\u093e\u0907\u0928": "emergency", # हेल्पलाइन
    # Tamil
    "\u0b9a\u0bbe\u0bb2\u0bc8": "road_authority",       # சாலை
    "\u0b95\u0bc1\u0bb4\u0bbf": "report_defect",        # குழி
    "\u0baa\u0bc1\u0b95\u0bbe\u0bb0\u0bcd": "report_defect", # புகார்
    "\u0baa\u0b9f\u0bcd\u0b9c\u0bc6\u0b9f\u0bcd": "budget", # பட்ஜெட்
    # Bengali
    "\u09b0\u09be\u09b8\u09cd\u09a4\u09be": "road_authority", # রাস্তা
    "\u0997\u09b0\u09cd\u09a4": "report_defect",        # গর্ত
    "\u0985\u09ad\u09bf\u09af\u09cb\u0997": "report_defect", # অভিযোগ
    "\u09ac\u09be\u099c\u09c7\u099f": "budget",          # বাজেট
    # Telugu
    "\u0c30\u0c4b\u0c21\u0c4d\u0c21\u0c41": "road_authority", # రోడ్డు
    "\u0c17\u0c41\u0c02\u0c24": "report_defect",        # గుంత
    "\u0c2b\u0c3f\u0c30\u0c4d\u0c2f\u0c3e\u0c26\u0c41": "report_defect", # ఫిర్యాదు
    "\u0c2c\u0c21\u0c4d\u0c1c\u0c46\u0c1f\u0c4d": "budget", # బడ్జెట్
    # Marathi (Devanagari)
    "\u0930\u0938\u094d\u0924\u093e": "road_authority",   # रस्ता
    "\u0916\u0921\u094d\u0921\u093e": "report_defect",    # खड्डा
    "\u0924\u0915\u094d\u0930\u093e\u0930": "report_defect", # तक्रार
    "\u092c\u091c\u0947\u091f": "budget",                # बजेट
}

# Unicode ranges for Indic scripts
_INDIC_SCRIPT_RANGES = {
    "Hindi":   (0x0900, 0x097F),   # Devanagari
    "Bengali": (0x0980, 0x09FF),
    "Tamil":   (0x0B80, 0x0BFF),
    "Telugu":  (0x0C00, 0x0C7F),
}


def _transliterate_to_english(text):
    """Replace Romanized Indic words with their English equivalents."""
    words = text.split()
    translated = []
    for w in words:
        clean = re.sub(r'[^a-zA-Z]', '', w).lower()
        if clean in INDIC_KEYWORDS:
            translated.append(INDIC_KEYWORDS[clean])
        else:
            translated.append(w)
    return " ".join(translated)


def _detect_native_script(text):
    """Detect native Indic script in text and return (language, topic) if found."""
    detected_lang = None
    for lang, (start, end) in _INDIC_SCRIPT_RANGES.items():
        for ch in text:
            if start <= ord(ch) <= end:
                detected_lang = lang
                break
        if detected_lang:
            break
    if not detected_lang:
        return None, None
    # Scan for known native-script keywords
    for keyword, topic in NATIVE_SCRIPT_KEYWORDS.items():
        if keyword in text:
            return detected_lang, topic
    return detected_lang, None


# ── Build TF-IDF Semantic Index ──
_corpus = []
_corpus_map = []
for _i, _entry in enumerate(KNOWLEDGE):
    _doc = " ".join(_entry["keys"]) + " " + _entry.get("topic", "") + " " + _entry["answer"][:300]
    _corpus.append(_doc)
    _corpus_map.append(_i)

_tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
_tfidf_matrix = _tfidf.fit_transform(_corpus)

_ALL_KEYWORDS = []
for _e in KNOWLEDGE:
    _ALL_KEYWORDS.extend(_e["keys"])

def _fuzzy_match_word(word, cutoff=0.7):
    if len(word) < 3:
        return None
    matches = get_close_matches(word, _ALL_KEYWORDS, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def _semantic_search(query, threshold=0.08):
    q_vec = _tfidf.transform([query])
    sims = cosine_similarity(q_vec, _tfidf_matrix)[0]
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    if best_score < threshold:
        return None, best_score
    return KNOWLEDGE[_corpus_map[best_idx]], best_score

# ── Intent Detection ──
INTENTS = [
    (re.compile(r"\b(who|which)\b.*(responsible|authority|department|handles|maintains|owns)", re.I), "road_authority"),
    (re.compile(r"\b(report|file|submit|lodge|register)\b.*(complaint|issue|defect|problem|pothole)", re.I), "report_defect"),
    (re.compile(r"\b(budget|spend|money|cost|fund|lakh|crore|expenditure|sanctioned|allocat)", re.I), "budget"),
    (re.compile(r"\b(emergency|accident|ambulance|police|fire|helpline|urgent call)", re.I), "emergency"),
    (re.compile(r"\b(rti|right to information|transparency act)", re.I), "rti"),
    (re.compile(r"\b(contractor|builder|grade|blacklist|accountability)", re.I), "contractor"),
    (re.compile(r"\b(pmgsy|pradhan mantri|gram sadak|5.year|five.year|maintenance warranty)", re.I), "pmgsy"),
    (re.compile(r"\b(hello|hi|hey|namaste|good morning|good evening|help me|what can you)", re.I), "greeting"),
    (re.compile(r"\b(severity|critical|urgent|sla|priority|how (long|soon))", re.I), "severity"),
    (re.compile(r"\b(lifecycle|lifespan|durability|how long.*(last|survive)|when.*(break|fail))", re.I), "lifecycle"),
    (re.compile(r"\b(how to use|guide|tutorial|step|instruction|navigate)", re.I), "guide"),
    (re.compile(r"\b(about|roadwatch|what is this|project|hackathon|feature)", re.I), "about"),
]

def _detect_intent(msg):
    for pattern, intent in INTENTS:
        if pattern.search(msg):
            return intent
    return None

def _find_by_topic(topic):
    for entry in KNOWLEDGE:
        if entry.get("topic") == topic:
            return entry
    return None

# Conversational prefixes for natural feel
_PREFIXES = {
    "road_authority": ["Great question! ", "Let me help you with that. ", "Here's the information on road authorities. "],
    "report_defect": ["I can definitely help you report that. ", "Let's get that issue logged. ", "Here's how to report the problem. "],
    "budget": ["Good question about transparency. ", "Here's what I know about road budgets. ", "Let me break down the budget information. "],
    "emergency": ["Here are the emergency numbers you need. ", "For immediate help, use these contacts. ", "Safety first! "],
    "greeting": ["", "", ""],
    "about": ["Thanks for your interest! ", "Let me tell you about RoadWatch. ", ""],
    "pmgsy": ["Important topic! ", "Here's what you need to know about PMGSY. ", ""],
    "contractor": ["Here's the contractor information. ", "Let me explain the grading system. ", ""],
    "rti": ["Great initiative! ", "Here's how to get that information. ", ""],
    "severity": ["Let me explain the severity system. ", "Good question! ", ""],
    "lifecycle": ["Interesting question! ", "Here's how road lifespan works. ", ""],
    "guide": ["Let me walk you through it. ", "Here's a quick overview. ", ""],
}

# Conversation memory
_conv_memory = defaultdict(list)

def smart_chat(msg, context=None, session_id="default", authority_map=None, lang=None):
    """Main chat entry point. Returns a natural, intelligent response."""
    low = msg.lower().strip()
    if not authority_map:
        authority_map = {}

    # ── AI Context Injection (LLM Pipeline) ──
    # If an LLM API key is present, use the advanced prompt wrapper for deep semantic processing
    if os.environ.get("LLM_API_KEY"):
        system_prompt = generate_llm_system_prompt(lang)
        llm_response = call_llm_api(system_prompt, msg, context or {}, authority_map)
        if llm_response:
            return llm_response
    # Otherwise, fallback to the local TF-IDF + Intent Engine

    # Handle follow-ups
    if any(x in low for x in ["tell me more", "more detail", "explain more", "elaborate", "go on", "continue", "expand"]):
        if _conv_memory.get(session_id):
            last_topic = _conv_memory[session_id][-1]
            entry = _find_by_topic(last_topic)
            if entry:
                return f"Sure, let me expand on that:\n\n{entry['answer']}"

    # Handle gratitude
    if any(x in low for x in ["thank", "thanks", "thx", "appreciate", "helpful", "great answer"]):
        return "You're welcome! Feel free to ask if you have any other questions about roads, authorities, or complaints. I'm here to help."

    # Handle yes/no
    if low in ["yes", "yeah", "yep", "sure", "ok", "okay"]:
        return "Great! What would you like to know more about? You can ask about road authorities, budget transparency, filing complaints, contractor grades, or emergency contacts."
    if low in ["no", "nope", "nah", "nothing", "no thanks"]:
        return "Alright! Let me know if you need anything else. I'm always here to help with road-related queries."

    # Location-aware responses
    if context and context.get("road"):
        road = context.get("road", "Unknown")
        rtype = context.get("road_type", "Unknown")
        district = context.get("district", "Unknown")
        state = context.get("state", "Unknown")
        auth = authority_map.get(rtype, {})
        loc_keywords = ["this road", "my road", "where am i", "current road", "here",
                        "responsible", "authority", "who", "which road", "what road",
                        "identify", "my location", "current location"]
        if any(k in low for k in loc_keywords):
            resp = f"Based on your GPS location, you are currently on:\n\n"
            resp += f"**{road}**\n"
            resp += f"Road Type: **{rtype}**\n"
            resp += f"District: **{district}**, {state}\n\n"
            if auth:
                resp += f"Responsible Authority: **{auth.get('authority', 'N/A')}**\n"
                resp += f"Officer: {auth.get('officer', 'N/A')}\n"
                resp += f"Helpline: **{auth.get('helpline', 'N/A')}**\n"
                resp += f"Portal: {auth.get('portal', 'N/A')}\n\n"
            if any(x in low for x in ["report", "complain", "defect", "pothole", "issue", "problem"]):
                resp += f"To report an issue on this road:\n"
                resp += f"1. Go to the **File Complaint** tab\n"
                resp += f"2. Your road name and type will be auto-filled from GPS\n"
                resp += f"3. Our AI classifies severity and routes to {auth.get('officer', 'the correct officer')}\n"
                resp += f"4. Resolution timeline depends on severity (48 hours to 30 days)"
            _conv_memory[session_id].append("road_authority")
            return resp

    # Step 0a: Native Indic-script detection (Devanagari, Bengali, Tamil, Telugu)
    native_lang, native_topic = _detect_native_script(msg)
    if native_lang and native_topic:
        entry = _find_by_topic(native_topic)
        if entry:
            _conv_memory[session_id].append(native_topic)
            lang_prefix = f"(Detected {native_lang} input) "
            return lang_prefix + entry["answer"]

    # Step 0b: Transliterate Romanized Indic words to English before processing
    msg_transliterated = _transliterate_to_english(low)
    if msg_transliterated != low:
        low = msg_transliterated
        msg = msg_transliterated

    # Step 1: Intent detection (handles natural phrasing like "who maintains highways?")
    intent = _detect_intent(msg)
    if intent:
        entry = _find_by_topic(intent)
        if entry:
            _conv_memory[session_id].append(intent)
            prefixes = _PREFIXES.get(intent, [""])
            prefix = prefixes[hash(msg) % len(prefixes)]
            answer = prefix + entry["answer"]
            if context and context.get("road"):
                answer += f"\n\n---\n*Your location: {context['road']} ({context.get('road_type','')}) | {context.get('district','')}*"
            return answer

    # Step 2: TF-IDF semantic search (handles paraphrasing and partial matches)
    corrected = low
    words = re.split(r'\s+', low)
    corrections = []
    for w in words:
        fix = _fuzzy_match_word(w)
        if fix and fix != w:
            corrected = corrected.replace(w, fix, 1)
            corrections.append((w, fix))

    entry_orig, score_orig = _semantic_search(low)
    entry_corr, score_corr = _semantic_search(corrected) if corrected != low else (None, 0)

    best_entry = None
    typo_note = ""
    if score_corr > score_orig and entry_corr:
        best_entry = entry_corr
        typo_note = "(I noticed some typos and auto-corrected: " + ", ".join(f'"{o}" to "{f}"' for o, f in corrections) + ")\n\n"
    elif entry_orig:
        best_entry = entry_orig

    if best_entry:
        topic = best_entry.get("topic", "general")
        _conv_memory[session_id].append(topic)
        prefixes = _PREFIXES.get(topic, [""])
        prefix = prefixes[hash(msg) % len(prefixes)]
        answer = typo_note + prefix + best_entry["answer"]
        if context and context.get("road"):
            answer += f"\n\n---\n*Your location: {context['road']} ({context.get('road_type','')}) | {context.get('district','')}*"
        return answer

    # Step 3: Nothing matched
    fallback = ("I wasn't able to find specific information for that query, but I'd love to help!\n\n"
            "Here are some things you can ask me about:\n"
            "- **Road authorities** - \"Who is responsible for NH roads?\"\n"
            "- **Filing complaints** - \"How do I report a pothole?\"\n"
            "- **Budget transparency** - \"How is road money spent?\"\n"
            "- **Contractor accountability** - \"What are contractor grades?\"\n"
            "- **Emergency contacts** - \"What's the NHAI helpline?\"\n"
            "- **PMGSY rules** - \"What is the 5-year maintenance rule?\"\n"
            "- **RTI filing** - \"How to file RTI for road data?\"\n\n"
            "You can also try rephrasing your question, or ask me something more specific!")
    if lang and lang != 'en':
        fallback = f"[Language: {lang}] " + fallback
    return fallback
