"""
PlacementIQ — Flask API
• Model trained on 80% of data (8000 rows)
• Test set (2000 rows) is strictly held-out — never seen during training
• /api/upload  — accepts CSV, runs bulk prediction, returns all rows with results
"""

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib, json, os, re, io
import requests as http_requests

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

# ── Load artifacts ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

model    = joblib.load(os.path.join(BASE, "placement_model.pkl"))
scaler   = joblib.load(os.path.join(BASE, "scaler.pkl"))
le_extra = joblib.load(os.path.join(BASE, "le_extra.pkl"))
le_train = joblib.load(os.path.join(BASE, "le_training.pkl"))
le_stat  = joblib.load(os.path.join(BASE, "le_status.pkl"))

with open(os.path.join(BASE, "model_stats.json")) as f:
    MODEL_STATS = json.load(f)

FEATURES = MODEL_STATS["features"]
PLACED_IDX = list(le_stat.classes_).index("Placed")

# Load main dataset (all 10k rows for the student browser)
_raw = pd.read_csv(os.path.join(BASE, "train.csv"))
df   = _raw.drop(columns=[c for c in _raw.columns if c.endswith(("_enc","_bin"))], errors="ignore")

# ── Column name aliases so uploaded CSVs with slightly different names work ──
COL_ALIASES = {
    "cgpa": "CGPA",
    "internship": "Internships", "internships": "Internships",
    "project": "Projects", "projects": "Projects",
    "workshop": "Workshops_Certifications",
    "workshops": "Workshops_Certifications",
    "workshops_certifications": "Workshops_Certifications",
    "certifications": "Workshops_Certifications",
    "aptitude": "AptitudeTestScore",
    "aptitudetestscore": "AptitudeTestScore",
    "aptitude_test_score": "AptitudeTestScore",
    "softskills": "SoftSkillsRating",
    "soft_skills": "SoftSkillsRating",
    "softskillsrating": "SoftSkillsRating",
    "soft_skills_rating": "SoftSkillsRating",
    "extracurricular": "ExtracurricularActivities",
    "extracurricularactivities": "ExtracurricularActivities",
    "extra_curricular": "ExtracurricularActivities",
    "placementtraining": "PlacementTraining",
    "placement_training": "PlacementTraining",
    "ssc": "SSC_Marks",
    "ssc_marks": "SSC_Marks",
    "sscmarks": "SSC_Marks",
    "hsc": "HSC_Marks",
    "hsc_marks": "HSC_Marks",
    "hscmarks": "HSC_Marks",
    "placementstatus": "PlacementStatus",
    "placement_status": "PlacementStatus",
    "status": "PlacementStatus",
}

REQUIRED_FEATURES = [
    "CGPA","Internships","Projects","Workshops_Certifications",
    "AptitudeTestScore","SoftSkillsRating",
    "ExtracurricularActivities","PlacementTraining",
    "SSC_Marks","HSC_Marks",
]


def _build_status_distribution(source_df: pd.DataFrame, status_values: pd.Series, col: str, values: list) -> dict:
    out = {"labels": [str(v) for v in values], "placed": [], "not_placed": []}
    for value in values:
        if isinstance(value, tuple):
            mask = (source_df[col] >= value[0]) & (source_df[col] < value[1])
        else:
            mask = source_df[col] == value
        status_slice = status_values[mask]
        out["placed"].append(int((status_slice == "Placed").sum()))
        out["not_placed"].append(int((status_slice == "NotPlaced").sum()))
    return out


def _build_dashboard_style_distributions(source_df: pd.DataFrame, status_values: pd.Series) -> dict:
    bins = [5, 6, 7, 7.5, 8, 8.5, 9, 10]
    labels = ["5-6", "6-7", "7-7.5", "7.5-8", "8-8.5", "8.5-9", "9-10"]
    cgpa_d = _build_status_distribution(
        source_df,
        status_values,
        "CGPA",
        list(zip(bins[:-1], bins[1:])),
    )
    cgpa_d["labels"] = labels

    apt_bins = list(range(40, 105, 5))
    apt_labels = [f"{x}-{x+5}" for x in apt_bins[:-1]]
    apt_d = _build_status_distribution(
        source_df,
        status_values,
        "AptitudeTestScore",
        list(zip(apt_bins[:-1], apt_bins[1:])),
    )
    apt_d["labels"] = apt_labels

    return {
        "cgpa_distribution": cgpa_d,
        "internship_distribution": _build_status_distribution(source_df, status_values, "Internships", [0, 1, 2, 3]),
        "projects_distribution": _build_status_distribution(source_df, status_values, "Projects", [0, 1, 2, 3, 4]),
        "aptitude_distribution": apt_d,
    }

# ── Helpers ─────────────────────────────────────────────────────────

def normalise_columns(upload_df: pd.DataFrame) -> pd.DataFrame:
    """Rename uploaded columns to canonical names using alias map."""
    rename = {}
    for col in upload_df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key in COL_ALIASES:
            rename[col] = COL_ALIASES[key]
    return upload_df.rename(columns=rename)


def encode_row(row: dict) -> tuple:
    """Encode one row dict → (label, prob_placed)."""
    extra_enc = 1 if str(row.get("ExtracurricularActivities","No")).strip().lower() in ("yes","1","true") else 0
    train_enc = 1 if str(row.get("PlacementTraining","No")).strip().lower() in ("yes","1","true") else 0

    arr = np.array([[
        float(row["CGPA"]),
        int(float(row["Internships"])),
        int(float(row["Projects"])),
        int(float(row["Workshops_Certifications"])),
        float(row["AptitudeTestScore"]),
        float(row["SoftSkillsRating"]),
        extra_enc, train_enc,
        float(row["SSC_Marks"]),
        float(row["HSC_Marks"]),
    ]])
    scaled = scaler.transform(arr)
    pred   = model.predict(scaled)[0]
    prob   = model.predict_proba(scaled)[0][PLACED_IDX]
    label  = le_stat.inverse_transform([pred])[0]
    return label, float(prob)


def bulk_predict(upload_df: pd.DataFrame) -> list:
    """Vectorised prediction for an entire uploaded dataframe."""
    cat_yes = {"yes","1","true"}

    extra_enc = upload_df["ExtracurricularActivities"].astype(str).str.strip().str.lower().isin(cat_yes).astype(int)
    train_enc = upload_df["PlacementTraining"].astype(str).str.strip().str.lower().isin(cat_yes).astype(int)

    X = pd.DataFrame({
        "CGPA":                        pd.to_numeric(upload_df["CGPA"],                       errors="coerce"),
        "Internships":                 pd.to_numeric(upload_df["Internships"],                errors="coerce").fillna(0).astype(int),
        "Projects":                    pd.to_numeric(upload_df["Projects"],                   errors="coerce").fillna(0).astype(int),
        "Workshops_Certifications":    pd.to_numeric(upload_df["Workshops_Certifications"],   errors="coerce").fillna(0).astype(int),
        "AptitudeTestScore":           pd.to_numeric(upload_df["AptitudeTestScore"],          errors="coerce"),
        "SoftSkillsRating":            pd.to_numeric(upload_df["SoftSkillsRating"],           errors="coerce"),
        "ExtracurricularActivities_enc": extra_enc,
        "PlacementTraining_enc":         train_enc,
        "SSC_Marks":                   pd.to_numeric(upload_df["SSC_Marks"],                  errors="coerce"),
        "HSC_Marks":                   pd.to_numeric(upload_df["HSC_Marks"],                  errors="coerce"),
    })

    # Flag rows with NaN
    bad_mask = X.isnull().any(axis=1)
    X_clean  = X.fillna(X.median(numeric_only=True))

    scaled      = scaler.transform(X_clean)
    preds       = model.predict(scaled)
    probs       = model.predict_proba(scaled)[:, PLACED_IDX]
    labels      = le_stat.inverse_transform(preds)

    results = []
    for i, (lbl, prob) in enumerate(zip(labels, probs)):
        row_out = upload_df.iloc[i].to_dict()
        # Cast numpy scalars
        for k, v in row_out.items():
            if hasattr(v, "item"):
                row_out[k] = v.item()
        row_out["_prediction"]   = lbl
        row_out["_probability"]  = round(float(prob) * 100, 1)
        row_out["_confidence"]   = ("High" if abs(prob - 0.5) > 0.25
                                    else "Medium" if abs(prob - 0.5) > 0.1
                                    else "Low")
        row_out["_parse_error"]  = bool(bad_mask.iloc[i])
        row_out["_placement_status_tag"] = "predictedYes" if lbl == "Placed" else "predictedNo"
        row_out["_suggestions"] = generate_suggestions(row_out, prob)
        row_out["_suggestion"] = row_out["_suggestions"][0] if row_out["_suggestions"] else ""

        # Accuracy check if ground truth exists
        if "PlacementStatus" in row_out and pd.notna(row_out["PlacementStatus"]):
            actual = str(row_out["PlacementStatus"]).strip()
            row_out["_actual"]  = actual
            row_out["_correct"] = (actual == lbl)
        results.append(row_out)

    return results


def generate_suggestions(data: dict, prob: float) -> list:
    tips = []
    cgpa        = float(data.get("CGPA", 0))
    internships = int(float(data.get("Internships", 0)))
    projects    = int(float(data.get("Projects", 0)))
    workshops   = int(float(data.get("Workshops_Certifications", 0)))
    aptitude    = float(data.get("AptitudeTestScore", 0))
    soft        = float(data.get("SoftSkillsRating", 0))
    extra       = str(data.get("ExtracurricularActivities","No"))
    training    = str(data.get("PlacementTraining","No"))

    tips.append("⭐ Excellent CGPA!" if cgpa >= 8 else "📈 Push CGPA above 8.0 for premium shortlists." if cgpa >= 7 else "📚 Boost CGPA above 7.0 — key recruiter filter.")
    tips.append("✅ Great internship count!" if internships >= 2 else "💼 1 internship is a start — target 2+ to boost odds." if internships == 1 else "💼 Zero internships — biggest gap. Apply on Internshala / LinkedIn now.")
    tips.append("🚀 Strong project portfolio!" if projects >= 3 else "🔨 Add more projects. Aim for 3–4 with diverse stacks." if projects >= 1 else "🔨 No projects. Build & host 2–3 on GitHub immediately.")
    tips.append("🎓 Good certifications!" if workshops >= 1 else "🎓 Earn 1–2 industry certs (AWS, Google, Coursera).")
    tips.append("🧠 Strong aptitude score!" if aptitude >= 75 else "🧠 Target 75+ aptitude. Practice on IndiaBix / HackerRank daily." if aptitude >= 65 else "🧠 Aptitude below average — 30 min daily practice on PrepInsta.")
    tips.append("🗣️ Excellent soft skills!" if soft >= 4 else "🗣️ Decent comm skills — practice STAR-format interview answers." if soft >= 3 else "🗣️ Soft skills need work — join mock interview groups / Toastmasters.")
    if extra.lower() not in ("yes","1","true"):
        tips.append("🏆 Join hackathons, open-source, or clubs — signals teamwork to recruiters.")
    if training.lower() not in ("yes","1","true"):
        tips.append("📋 Enroll in a placement training program — mock interviews significantly raise odds.")
    tips.append("🎯 TOP tier candidate — focus on DSA, system design, company-specific prep." if prob >= 0.8 else "📊 Above average — fix weakest 2 areas to be highly competitive." if prob >= 0.6 else "⚡ Quick wins: 1 internship + 2 projects + aptitude practice → big improvement in 3 months.")
    return tips


def analyze_github(username: str) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN","")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    u = http_requests.get(f"https://api.github.com/users/{username}", headers=headers, timeout=10)
    if u.status_code != 200:
        return {"error": f"GitHub user '{username}' not found"}
    user = u.json()

    r = http_requests.get(f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated", headers=headers, timeout=10)
    repos = r.json() if r.status_code == 200 else []
    if not isinstance(repos, list): repos = []

    langs, topics = {}, []
    for repo in repos:
        if repo.get("language"): langs[repo["language"]] = langs.get(repo["language"],0)+1
        topics.extend(repo.get("topics",[]))

    stars = sum(x.get("stargazers_count",0) for x in repos)
    forks = sum(x.get("forks_count",0) for x in repos)
    n     = len(repos)
    score = round((min(stars/50,1)*.3 + min(n/20,1)*.3 + min(len(langs)/5,1)*.2 + min(user.get("followers",0)/100,1)*.2)*100, 1)

    improvements = []
    if n < 5:           improvements.append("📁 Create more repos — aim for 10+ to show consistent activity.")
    if stars < 5:       improvements.append("⭐ Focus on quality projects that solve real problems to earn stars.")
    if len(langs) < 3:  improvements.append("💻 Diversify tech stack — multiple languages signals versatility.")
    if not user.get("bio"):  improvements.append("📝 Add a professional bio with your skills and interests.")
    if not user.get("blog"): improvements.append("🌐 Link your portfolio or LinkedIn to your profile.")
    desc_count = sum(1 for repo in repos if repo.get("description"))
    if n > 0 and desc_count < n*0.5: improvements.append("📄 Add descriptions to all repos — empty ones look unprofessional.")
    if not any(repo.get("name","").lower() == username.lower() for repo in repos):
        improvements.append("🏠 Create a profile README (repo named same as username) to stand out.")

    top_repos = sorted(repos, key=lambda x: -(x.get("stargazers_count",0)+x.get("forks_count",0)))[:5]
    return {
        "username": username, "name": user.get("name", username),
        "avatar_url": user.get("avatar_url",""), "bio": user.get("bio",""),
        "location": user.get("location",""),
        "followers": user.get("followers",0), "following": user.get("following",0),
        "public_repos": n, "total_stars": stars, "total_forks": forks,
        "github_score": score,
        "top_languages": [{"lang":l,"count":c} for l,c in sorted(langs.items(),key=lambda x:-x[1])[:5]],
        "top_repos": [{"name":r["name"],"description":r.get("description",""),
                       "stars":r.get("stargazers_count",0),"forks":r.get("forks_count",0),
                       "language":r.get("language",""),"url":r.get("html_url",""),
                       "updated":r.get("updated_at","")[:10]} for r in top_repos],
        "topics": list(set(topics))[:15],
        "improvements": improvements,
        "profile_url": f"https://github.com/{username}",
    }


def predict_from_github(github_analysis: dict) -> dict:
    """
    Convert GitHub profile metrics to placement prediction features.
    Maps GitHub activity to estimated feature values for placement model.
    """
    if "error" in github_analysis:
        return {"error": github_analysis["error"]}

    # Extract GitHub metrics
    repos = github_analysis.get("public_repos", 0)
    stars = github_analysis.get("total_stars", 0)
    forks = github_analysis.get("total_forks", 0)
    langs = len(github_analysis.get("top_languages", []))
    followers = github_analysis.get("followers", 0)
    gh_score = github_analysis.get("github_score", 0)

    # Map GitHub metrics to placement features
    # Projects: repos indicate project count
    projects = min(max(repos - 1, 0), 4)  # Cap at 4

    # Internships: estimate from stars and forks (quality indicator)
    internships = min(max(int((stars + forks) / 20), 0), 3)  # Cap at 3

    # Workshops/Certifications: languages diversity
    workshops = min(max(langs - 1, 0), 3)  # Cap at 3

    # AptitudeTestScore: map from GitHub score (40-100 range)
    aptitude_score = 40 + (gh_score / 100) * 60  # 40 + (0-60) based on gh_score

    # SoftSkillsRating: followers as community engagement (0-5 scale)
    soft_skills = min(max(followers / 20, 0), 5)  # Normalize followers to 0-5

    # Extracurricular: have meaningful GitHub presence
    extracurricular = "Yes" if repos >= 5 and gh_score >= 50 else "No"

    # PlacementTraining: have structured portfolio (profile README, etc.)
    training = "Yes" if repos >= 10 and len(github_analysis.get("topics", [])) >= 3 else "No"

    # Estimated scores (use training set averages for missing data)
    cgpa = 7.5  # Default CGPA
    ssc_marks = 80  # Default SSC
    hsc_marks = 85  # Default HSC

    # Build feature dict for prediction
    features = {
        "CGPA": cgpa,
        "Internships": int(internships),
        "Projects": int(projects),
        "Workshops_Certifications": int(workshops),
        "AptitudeTestScore": round(aptitude_score, 2),
        "SoftSkillsRating": round(soft_skills, 2),
        "ExtracurricularActivities": extracurricular,
        "PlacementTraining": training,
        "SSC_Marks": ssc_marks,
        "HSC_Marks": hsc_marks,
    }

    # Get prediction
    label, prob = encode_row(features)

    return {
        "prediction": label,
        "probability": round(prob * 100, 1),
        "confidence": "High" if abs(prob - 0.5) > 0.25 else "Medium" if abs(prob - 0.5) > 0.1 else "Low",
        "estimated_features": {
            "CGPA": features["CGPA"],
            "Internships": features["Internships"],
            "Projects": features["Projects"],
            "Workshops_Certifications": features["Workshops_Certifications"],
            "AptitudeTestScore": features["AptitudeTestScore"],
            "SoftSkillsRating": features["SoftSkillsRating"],
            "ExtracurricularActivities": features["ExtracurricularActivities"],
            "PlacementTraining": features["PlacementTraining"],
        },
        "suggestions": generate_suggestions(features, prob),
        "github_improvements": github_analysis.get("improvements", []),
    }


# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health")
def health():
    return jsonify({"status":"ok", "model_accuracy": MODEL_STATS["model_accuracy"],
                    "train_size": MODEL_STATS.get("train_size"),
                    "test_size":  MODEL_STATS.get("test_size")})

@app.route("/api/stats")
def get_stats():
    placed_df     = df[df["PlacementStatus"]=="Placed"]
    not_placed_df = df[df["PlacementStatus"]=="NotPlaced"]

    def dist(col, vals):
        out = {"labels":[str(v) for v in vals],"placed":[],"not_placed":[]}
        for v in vals:
            sub = df[df[col]==v] if not isinstance(v,tuple) else df[(df[col]>=v[0])&(df[col]<v[1])]
            out["placed"].append(int((sub["PlacementStatus"]=="Placed").sum()))
            out["not_placed"].append(int((sub["PlacementStatus"]=="NotPlaced").sum()))
        return out

    bins   = [5,6,7,7.5,8,8.5,9,10]
    labels = ["5-6","6-7","7-7.5","7.5-8","8-8.5","8.5-9","9-10"]
    cgpa_d = {"labels":labels,"placed":[],"not_placed":[]}
    for i,lbl in enumerate(labels):
        sub = df[(df["CGPA"]>=bins[i])&(df["CGPA"]<bins[i+1])]
        cgpa_d["placed"].append(int((sub["PlacementStatus"]=="Placed").sum()))
        cgpa_d["not_placed"].append(int((sub["PlacementStatus"]=="NotPlaced").sum()))

    apt_bins   = list(range(40,105,5))
    apt_labels = [f"{x}-{x+5}" for x in apt_bins[:-1]]
    apt_d      = {"labels":apt_labels,"placed":[],"not_placed":[]}
    for i in range(len(apt_labels)):
        sub = df[(df["AptitudeTestScore"]>=apt_bins[i])&(df["AptitudeTestScore"]<apt_bins[i+1])]
        apt_d["placed"].append(int((sub["PlacementStatus"]=="Placed").sum()))
        apt_d["not_placed"].append(int((sub["PlacementStatus"]=="NotPlaced").sum()))

    cm = MODEL_STATS.get("confusion_matrix",[[0,0],[0,0]])
    return jsonify({
        "summary": {
            "total": len(df),
            "placed": int((df["PlacementStatus"]=="Placed").sum()),
            "not_placed": int((df["PlacementStatus"]=="NotPlaced").sum()),
            "placement_rate": round((df["PlacementStatus"]=="Placed").mean()*100,1),
            "avg_cgpa_placed": round(placed_df["CGPA"].mean(),2),
            "avg_cgpa_not_placed": round(not_placed_df["CGPA"].mean(),2),
            "model_accuracy": round(MODEL_STATS["model_accuracy"]*100,1),
            "model_f1": round(MODEL_STATS.get("model_f1",0)*100,1),
            "model_roc_auc": round(MODEL_STATS.get("model_roc_auc",0)*100,1),
            "train_size": MODEL_STATS.get("train_size",8000),
            "test_size":  MODEL_STATS.get("test_size",2000),
            "cv_mean": round(MODEL_STATS.get("cv_mean",0)*100,1),
            "cv_std":  round(MODEL_STATS.get("cv_std",0)*100,1),
            "confusion_matrix": cm,
            "feature_importance": MODEL_STATS["feature_importance"],
        },
        "cgpa_distribution":      cgpa_d,
        "internship_distribution": dist("Internships",[0,1,2,3]),
        "projects_distribution":   dist("Projects",[0,1,2,3,4]),
        "aptitude_distribution":   apt_d,
    })

@app.route("/api/students")
def get_students():
    page     = int(request.args.get("page",1))
    per_page = int(request.args.get("per_page",20))
    status   = request.args.get("status","")
    search   = request.args.get("search","").strip()
    sort_by  = request.args.get("sort_by","CGPA")
    order    = request.args.get("order","desc")

    filtered = df.copy()
    if status in ["Placed","NotPlaced"]:
        filtered = filtered[filtered["PlacementStatus"]==status]
    if search:
        filtered = filtered[filtered["StudentID"].str.contains(search,case=False,na=False)]
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=(order=="asc"))

    total = len(filtered)
    page_df = filtered.iloc[(page-1)*per_page : page*per_page]

    COLS = ["StudentID","CGPA","Internships","Projects","Workshops_Certifications",
            "AptitudeTestScore","SoftSkillsRating","ExtracurricularActivities",
            "PlacementTraining","SSC_Marks","HSC_Marks","PlacementStatus"]
    out = page_df[COLS].copy()
    out["Internships"] = out["Internships"].astype(int)
    out["Projects"]    = out["Projects"].astype(int)
    out["Workshops_Certifications"] = out["Workshops_Certifications"].astype(int)

    return jsonify({"students": out.to_dict(orient="records"),
                    "total": total, "page": page, "per_page": per_page,
                    "total_pages": max(1,(total+per_page-1)//per_page)})

@app.route("/api/student/<sid>")
def get_student(sid):
    row = df[df["StudentID"]==sid]
    if row.empty: return jsonify({"error":"Student not found"}),404
    record = {k:(v.item() if hasattr(v,"item") else v) for k,v in row.iloc[0].to_dict().items()}
    label, prob = encode_row(record)
    pcts = {col: round((df[col]<record[col]).mean()*100,1)
            for col in ["CGPA","AptitudeTestScore","SoftSkillsRating","SSC_Marks","HSC_Marks"]}
    return jsonify({"student": record,
                    "prediction": {"label":label,"probability":prob},
                    "suggestions": generate_suggestions(record,prob),
                    "percentiles": pcts})

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json or {}
    missing = [k for k in REQUIRED_FEATURES if k not in data]
    if missing: return jsonify({"error":f"Missing: {missing}"}),400
    label, prob = encode_row(data)
    return jsonify({"prediction":label,"probability":prob,
                    "suggestions": generate_suggestions(data,prob),
                    "confidence":"High" if abs(prob-.5)>.25 else "Medium" if abs(prob-.5)>.1 else "Low"})

# ── CSV UPLOAD ────────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_csv():
    """
    Accept a CSV file upload.
    • Normalises column names via aliases
    • Runs vectorised bulk prediction
    • If PlacementStatus column present → computes per-row accuracy + summary metrics
    • Returns JSON with results array + metadata
    """
    if "file" not in request.files:
        return jsonify({"error":"No file uploaded. Send a multipart form with field 'file'."}),400

    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error":"Only CSV files are accepted."}),400

    try:
        content = f.read().decode("utf-8", errors="replace")
        upload_df = pd.read_csv(io.StringIO(content))
    except Exception as e:
        return jsonify({"error":f"Could not parse CSV: {str(e)}"}),400

    # Normalise columns
    upload_df = normalise_columns(upload_df)

    # Check required columns
    missing = [c for c in REQUIRED_FEATURES if c not in upload_df.columns]
    if missing:
        sample_cols = list(upload_df.columns)
        return jsonify({
            "error": f"CSV is missing required columns: {missing}",
            "your_columns": sample_cols,
            "required_columns": REQUIRED_FEATURES,
            "hint": "Column names are case-insensitive. Aliases like 'cgpa', 'aptitude', 'ssc' are accepted."
        }), 400

    total_rows = len(upload_df)
    if total_rows == 0:
        return jsonify({"error":"CSV is empty."}),400
    if total_rows > 5000:
        return jsonify({"error":f"CSV has {total_rows} rows. Max allowed is 5000 per upload."}),400

    results = bulk_predict(upload_df)
    pred_status = pd.Series([r["_prediction"] for r in results], index=upload_df.index)
    upload_distributions = _build_dashboard_style_distributions(upload_df, pred_status)

    # Aggregate metrics
    has_ground_truth = "_actual" in results[0]
    placed_count     = sum(1 for r in results if r["_prediction"]=="Placed")
    errors           = sum(1 for r in results if r.get("_parse_error"))

    meta = {
        "total_rows":    total_rows,
        "predicted_placed":    placed_count,
        "predicted_notplaced": total_rows - placed_count,
        "prediction_rate":     round(placed_count/total_rows*100,1),
        "parse_errors":        errors,
        "has_ground_truth":    has_ground_truth,
    }

    if has_ground_truth:
        correct    = sum(1 for r in results if r.get("_correct"))
        tp = sum(1 for r in results if r["_prediction"]=="Placed"     and r.get("_actual")=="Placed")
        tn = sum(1 for r in results if r["_prediction"]=="NotPlaced"  and r.get("_actual")=="NotPlaced")
        fp = sum(1 for r in results if r["_prediction"]=="Placed"     and r.get("_actual")=="NotPlaced")
        fn = sum(1 for r in results if r["_prediction"]=="NotPlaced"  and r.get("_actual")=="Placed")
        meta.update({
            "accuracy":    round(correct/total_rows*100,2),
            "correct":     correct,
            "incorrect":   total_rows - correct,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision":   round(tp/(tp+fp)*100,2) if (tp+fp)>0 else 0,
            "recall":      round(tp/(tp+fn)*100,2) if (tp+fn)>0 else 0,
        })
        prec = meta["precision"]/100
        rec  = meta["recall"]/100
        meta["f1"] = round(2*prec*rec/(prec+rec)*100,2) if (prec+rec)>0 else 0

    charts = {
        **upload_distributions,
        "feature_importance": MODEL_STATS["feature_importance"],
    }
    if has_ground_truth:
        charts["confusion_matrix"] = [[meta["tn"], meta["fp"]], [meta["fn"], meta["tp"]]]

    return jsonify({"meta": meta, "results": results, "charts": charts})


@app.route("/api/github/<username>")
def github_analysis(username):
    if not re.match(r"^[a-zA-Z0-9\-]{1,39}$", username):
        return jsonify({"error":"Invalid GitHub username"}),400
    result = analyze_github(username)
    return jsonify(result), (404 if "error" in result else 200)


@app.route("/api/github-predict/<username>")
def github_predict(username):
    """
    Predict placement based on GitHub profile analysis.
    • Fetches GitHub metrics
    • Maps to placement features
    • Returns prediction + suggestions + GitHub improvements
    """
    if not re.match(r"^[a-zA-Z0-9\-]{1,39}$", username):
        return jsonify({"error":"Invalid GitHub username"}),400

    github_data = analyze_github(username)
    if "error" in github_data:
        return jsonify(github_data), 404

    prediction = predict_from_github(github_data)

    return jsonify({
        "github_profile": {
            "username": github_data.get("username"),
            "name": github_data.get("name"),
            "avatar_url": github_data.get("avatar_url"),
            "bio": github_data.get("bio"),
            "location": github_data.get("location"),
            "followers": github_data.get("followers"),
            "following": github_data.get("following"),
            "public_repos": github_data.get("public_repos"),
            "total_stars": github_data.get("total_stars"),
            "total_forks": github_data.get("total_forks"),
            "github_score": github_data.get("github_score"),
            "top_languages": github_data.get("top_languages"),
            "top_repos": github_data.get("top_repos"),
            "topics": github_data.get("topics"),
            "profile_url": github_data.get("profile_url"),
        },
        "placement_prediction": {
            "prediction": prediction["prediction"],
            "probability": prediction["probability"],
            "confidence": prediction["confidence"],
            "estimated_features": prediction["estimated_features"],
        },
        "all_suggestions": {
            "placement_strategies": prediction["suggestions"],
            "github_improvements": prediction["github_improvements"],
        }
    })



if __name__ == "__main__":
    import threading, webbrowser, time
    def _open():
        time.sleep(1.2)
        webbrowser.open("http://localhost:5000")
    threading.Thread(target=_open, daemon=True).start()
    print("\n  ✅  PlacementIQ → http://localhost:5000\n")
    app.run(debug=False, port=5000, host="0.0.0.0")
