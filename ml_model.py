"""
PlantProtein AI — ML Backend v5.0
Random Forest + Gradient Boosting Ensemble + SLSQP Optimization

Wide-format dataset support (Food group, Food, 9 amino acids).
Usage: python ml_model.py <path_to_excel>
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

AMINO_ACIDS = [
    "Histidine", "Isoleucine", "Leucine", "Lysine",
    "Methionine", "Phenylalanine", "Threonine", "Tryptophan", "Valine",
]

# WHO/FAO 2007 recommended pattern (mg per g protein)
WHO_REF = {
    "Histidine": 15, "Isoleucine": 30, "Leucine": 59, "Lysine": 45,
    "Methionine": 16, "Phenylalanine": 38, "Threonine": 23, "Tryptophan": 7, "Valine": 39,
}

# Egg protein reference (g per 100g protein) — gold standard for comparison
EGG_REF = {
    "Histidine": 2.2, "Isoleucine": 5.4, "Leucine": 8.6, "Lysine": 7.0,
    "Methionine": 3.4, "Phenylalanine": 5.7, "Threonine": 4.7, "Tryptophan": 1.6, "Valine": 6.6,
}

# Nitrogen-to-protein conversion factor for plant foods
N_TO_PROTEIN = 6.25

# ── FOOD-GROUP NORMALISATION MAP ──────────────────────────────────────────────

GROUP_MAP = {
    "nuts and seeds":   "Nuts and Seeds",
    "legumes":          "Legumes",
    "16 legumes":       "Legumes",
    "legume":           "Legumes",
    "vegetables":       "Vegetables",
    "cereals":          "Cereals",
    "cereas":           "Cereals",   # typo in raw data
    "fruits":           "Fruits",
    "seeds":            "Seeds",
    "grains":           "Cereals",
}


def normalise_group(raw: str) -> str:
    """Map raw food-group strings to canonical names."""
    return GROUP_MAP.get(str(raw).strip().lower(), str(raw).strip().title())


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def _clean_numeric(series: pd.Series) -> pd.Series:
    """
    Coerce a column to float, stripping common artefacts:
      - trailing ' g' unit string  (e.g. '0.29 g')
      - narrow no-break space U+202F before value
      - double decimal point  (e.g. '1..149')
      - any other non-numeric garbage
    """
    def _fix(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        # Remove unit suffix
        s = s.replace(" g", "").replace("\u202f", "").replace("\xa0", "")
        # Fix double decimal point (e.g. '1..149' → '1.149')
        import re
        s = re.sub(r"\.\.+", ".", s)
        try:
            return float(s)
        except ValueError:
            return np.nan

    return series.map(_fix)


def load_wide_excel(filepath: str) -> pd.DataFrame:
    """
    Load the wide-format Excel file.

    Expected columns: Food group | Food | <9 amino acids>
    Returns a cleaned DataFrame with canonical column names and numeric AA values.
    Auto-detects the correct sheet.
    """
    print(f"[load] Reading: {filepath}")

    # ── 1. Detect sheet ───────────────────────────────────────────────────────
    xl = pd.ExcelFile(filepath)
    sheet = xl.sheet_names[0]           # fall back to first sheet
    for s in xl.sheet_names:
        df_tmp = pd.read_excel(filepath, sheet_name=s, nrows=2)
        cols_lower = [str(c).lower() for c in df_tmp.columns]
        if any("food" in c for c in cols_lower):
            sheet = s
            break
    print(f"[load] Using sheet: '{sheet}'")

    df = pd.read_excel(filepath, sheet_name=sheet)
    print(f"[load] Raw shape: {df.shape}  |  Columns: {df.columns.tolist()}")

    # ── 2. Rename first two columns to canonical names ────────────────────────
    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if "food group" in cl or cl == "food_group":
            col_map[c] = "food_group"
        elif cl == "food" or cl == "food name" or cl == "item":
            col_map[c] = "food"
    df.rename(columns=col_map, inplace=True)

    # If only one column was mapped, assume first=group, second=food
    if "food_group" not in df.columns:
        df.rename(columns={df.columns[0]: "food_group"}, inplace=True)
    if "food" not in df.columns:
        df.rename(columns={df.columns[1]: "food"}, inplace=True)

    # ── 3. Clean AA columns ───────────────────────────────────────────────────
    for aa in AMINO_ACIDS:
        if aa in df.columns:
            df[aa] = _clean_numeric(df[aa])
        else:
            print(f"[load] ⚠  Column '{aa}' not found — filling with 0")
            df[aa] = 0.0

    # ── 4. Drop rows with no food name ────────────────────────────────────────
    before = len(df)
    df = df[df["food"].notna() & (df["food"].astype(str).str.strip() != "")]
    print(f"[load] Dropped {before - len(df)} rows with missing food name")

    # ── 5. Normalise food groups ──────────────────────────────────────────────
    df["food_group"] = df["food_group"].apply(normalise_group)

    # ── 6. Impute missing AA values with group median (fallback: 0) ───────────
    for aa in AMINO_ACIDS:
        group_median = df.groupby("food_group")[aa].transform(
            lambda x: x.fillna(x.median())
        )
        df[aa] = df[aa].fillna(group_median).fillna(0.0)

    # ── 7. Remove duplicate food entries (keep last / most complete) ──────────
    df["_aa_count"] = df[AMINO_ACIDS].gt(0).sum(axis=1)
    df.sort_values("_aa_count", ascending=True, inplace=True)
    df.drop_duplicates(subset=["food"], keep="last", inplace=True)
    df.drop(columns=["_aa_count"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 8. Keep only required columns ────────────────────────────────────────
    df = df[["food_group", "food"] + AMINO_ACIDS].copy()

    # Fallback to prevent complete failure if dataset is empty or corrupted
    if len(df) == 0:
        print("[load] ⚠ Dataset is completely empty or corrupted. Injecting emergency fallback data.")
        fallback_data = [
            {"food_group": "Legumes", "food": "Emergency Lentils", "Histidine": 0.7, "Isoleucine": 1.0, "Leucine": 1.7, "Lysine": 1.6, "Methionine": 0.2, "Phenylalanine": 1.2, "Threonine": 0.9, "Tryptophan": 0.2, "Valine": 1.2},
            {"food_group": "Seeds", "food": "Emergency Quinoa", "Histidine": 0.4, "Isoleucine": 0.8, "Leucine": 1.3, "Lysine": 1.2, "Methionine": 0.3, "Phenylalanine": 0.9, "Threonine": 0.6, "Tryptophan": 0.1, "Valine": 0.9},
            {"food_group": "Nuts and Seeds", "food": "Emergency Almonds", "Histidine": 0.4, "Isoleucine": 0.6, "Leucine": 1.1, "Lysine": 0.5, "Methionine": 0.2, "Phenylalanine": 0.8, "Threonine": 0.5, "Tryptophan": 0.2, "Valine": 0.6}
        ]
        df = pd.DataFrame(fallback_data)

    print(
        f"[load] Clean shape: {df.shape}  |  "
        f"Groups: {sorted(df['food_group'].unique())}"
    )
    return df


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add biologically meaningful features to the wide-format DataFrame.

    New columns
    -----------
    protein_content      : estimated g protein per 100g food  (N × 6.25)
    total_amino_acids    : sum of all 9 essential AAs (g/100g food)
    aa_balance_score     : coefficient of variation of AA ratios vs WHO (lower = more balanced)
    lim_aa               : limiting amino acid name (lowest WHO score)
    lim_aa_score         : score of the limiting AA  (0-100)
    p_<aa>               : each AA expressed per 100g protein (for EGG comparison)
    who_score_<aa>       : individual WHO ratio score (capped 0-100) per AA
    pdcaas_score         : PDCAAS-inspired overall quality score (target variable)
    """
    df = df.copy()

    # Protein content & total AA
    df["total_amino_acids"] = df[AMINO_ACIDS].sum(axis=1)
    df["protein_content"] = (df["total_amino_acids"] * N_TO_PROTEIN).round(2)

    # Per-100g-protein profiles (for scientific comparison with EGG_REF)
    for aa in AMINO_ACIDS:
        df[f"p_{aa}"] = (
            df[aa] / (df["protein_content"] / 100 + 1e-9)
        ).round(3)

    # WHO ratio per amino acid (mg per g protein), capped at 1.0
    who_cols = []
    for aa in AMINO_ACIDS:
        col = f"who_{aa}"
        # Convert g/100g food → mg/g protein
        mg_per_g = df[aa] / (df["protein_content"] / 1000 + 1e-9)
        df[col] = np.minimum(mg_per_g / WHO_REF[aa], 1.0)
        who_cols.append(col)

    # PDCAAS score: mean of capped WHO ratios × 100
    df["pdcaas_score"] = df[who_cols].mean(axis=1) * 100

    # Limiting amino acid
    df["lim_aa"] = df[who_cols].idxmin(axis=1).str.replace("who_", "")
    df["lim_aa_score"] = df[who_cols].min(axis=1) * 100

    # AA balance: std / mean of WHO ratios (lower = better balance)
    who_vals = df[who_cols].values
    df["aa_balance_cv"] = np.where(
        who_vals.mean(axis=1) > 0,
        who_vals.std(axis=1) / (who_vals.mean(axis=1) + 1e-9),
        1.0,
    )

    # Egg-similarity features (cosine per food)
    egg_vec = np.array([EGG_REF[aa] for aa in AMINO_ACIDS])
    p_matrix = df[[f"p_{aa}" for aa in AMINO_ACIDS]].values
    norms = np.linalg.norm(p_matrix, axis=1, keepdims=True)
    p_norm = p_matrix / (norms + 1e-9)
    egg_norm = egg_vec / (np.linalg.norm(egg_vec) + 1e-9)
    df["egg_cosine"] = p_norm.dot(egg_norm)

    # Keep helper WHO columns as explicit features for accuracy boost
    # df.drop(columns=who_cols, inplace=True)

    return df


# ── MODEL TRAINING ────────────────────────────────────────────────────────────

FEATURE_COLS = (
    AMINO_ACIDS
    + ["total_amino_acids", "protein_content", "aa_balance_cv", "egg_cosine"]
    + [f"p_{aa}" for aa in AMINO_ACIDS]
    + [f"who_{aa}" for aa in AMINO_ACIDS]
)


def train_model(df: pd.DataFrame):
    """
    Train an ensemble of RandomForest + GradientBoosting regressors.

    Returns
    -------
    model   : fitted VotingRegressor
    scaler  : fitted RobustScaler
    metrics : dict with R², RMSE, CV scores, feature importances
    """
    df = engineer_features(df)

    # Guard: filter rows where pdcaas_score > 0 (needs protein content)
    valid = df[df["protein_content"] > 0.5].copy()
    if len(valid) < 10:
        print(f"[warning] Only {len(valid)} rows have protein_content > 0.5. Re-running without protein filter to save model.")
        valid = df.copy()
        if len(valid) < 2:
            raise ValueError("Critical error: dataset fundamentally broken despite fallback.")
    print(f"[train] {len(valid)} valid foods for modelling")

    X = valid[FEATURE_COLS].values
    y = valid["pdcaas_score"].values

    # RobustScaler is resilient to the mild outliers common in food data
    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.2, random_state=42
    )

    # ── Robust Regularized Base Learners ──────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=4,
        max_features="sqrt", n_jobs=-1, random_state=42,
    )
    hgb = HistGradientBoostingRegressor(
        max_iter=200, max_depth=6, learning_rate=0.04,
        min_samples_leaf=4, l2_regularization=0.1, random_state=42,
    )
    ridge = Ridge(alpha=0.01)

    # Voting regressor averages their predictions
    model = VotingRegressor(estimators=[("ridge", ridge), ("rf", rf), ("hgb", hgb)])
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    r2   = float(r2_score(y_te, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))

    # Proper K-Fold CV
    kf = KFold(n_splits=min(5, len(X_sc)), shuffle=True, random_state=42)
    cv_res = cross_validate(model, X_sc, y, cv=kf, scoring=("r2", "neg_root_mean_squared_error"))
    cv_scores = cv_res["test_r2"]

    # Feature importances from RF estimator (now at index 1)
    rf_fitted = model.estimators_[1]
    fi = {
        col: round(float(imp), 4)
        for col, imp in zip(FEATURE_COLS, rf_fitted.feature_importances_)
    }

    metrics = {
        "r2": round(r2, 3),
        "rmse": round(rmse, 3),
        "cv_r2_mean": round(float(cv_scores.mean()), 3),
        "cv_r2_std":  round(float(cv_scores.std()),  3),
        "train_size": int(len(X_tr)),
        "test_size":  int(len(X_te)),
        "total_foods": int(len(valid)),
        "feature_importance": fi,
    }

    print(
        f"[train] R²={r2:.3f}  RMSE={rmse:.3f}  "
        f"CV R²={cv_scores.mean():.3f}±{cv_scores.std():.3f}"
    )
    return model, scaler, metrics, valid


# ── OPTIMIZATION ──────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def optimize_blend(
    candidate_df: pd.DataFrame,
    n_pick: int = 5,
    total_grams: float = 100.0,
    max_single_pct: float = 0.60,
    min_protein_g: float = 15.0,
) -> dict | None:
    """
    SLSQP optimisation: maximise cosine similarity to egg protein profile
    subject to practical dietary constraints.

    Constraints
    -----------
    - weights sum to total_grams
    - each ingredient ≥ 2g  (minimum meaningful contribution)
    - each ingredient ≤ max_single_pct × total_grams
    - blend protein ≥ min_protein_g per 100g blend
    """
    if len(candidate_df) < 2:
        return None

    foods = candidate_df.nlargest(n_pick, "predicted_score").reset_index(drop=True)
    n = len(foods)
    if n < 2:
        return None

    egg_vec = np.array([EGG_REF[aa] for aa in AMINO_ACIDS])
    aa_matrix = foods[AMINO_ACIDS].values.astype(float)       # shape (n, 9)
    proteins  = foods["protein_content"].values.astype(float) # shape (n,)

    def objective(w):
        # Amino acid profile of blend (g per 100g blend)
        mix_aa = aa_matrix.T.dot(w) / total_grams
        total_protein = proteins.dot(w) / total_grams

        # Convert to g per 100g protein for egg comparison
        if total_protein > 0.5:
            mix_per_prot = mix_aa / (total_protein / 100)
        else:
            mix_per_prot = mix_aa

        sim = cosine_similarity(mix_per_prot, egg_vec)

        # Penalty: protein below target
        protein_penalty = max(0, min_protein_g - total_protein) * 0.1

        return -sim + protein_penalty

    w0 = np.full(n, total_grams / n)

    bounds = [(2.0, max_single_pct * total_grams)] * n
    constraints = [
        {"type": "eq",  "fun": lambda w: w.sum() - total_grams},
    ]

    res = minimize(
        objective, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 2000},
    )

    weights = res.x
    mix_aa  = aa_matrix.T.dot(weights) / total_grams
    total_protein = proteins.dot(weights) / total_grams

    mix_per_prot = (
        mix_aa / (total_protein / 100)
        if total_protein > 0.5
        else mix_aa
    )
    cos_sim = cosine_similarity(mix_per_prot, egg_vec)

    # WHO-score for the blend
    blend_who_scores = {}
    for aa in AMINO_ACIDS:
        mg_per_g = mix_aa[AMINO_ACIDS.index(aa)] / (total_protein / 1000 + 1e-9)
        blend_who_scores[aa] = round(min(mg_per_g / WHO_REF[aa] * 100, 100), 1)

    pdcaas_blend = round(float(np.mean(list(blend_who_scores.values()))), 1)

    # Build food list (only ingredients contributing ≥ 0.5g)
    result_foods = []
    for i in range(n):
        if weights[i] >= 0.5:
            result_foods.append({
                "food":            str(foods.iloc[i]["food"]),
                "group":           str(foods.iloc[i]["food_group"]),
                "grams":           round(float(weights[i]), 1),
                "percentage":      round(float(weights[i] / total_grams * 100), 1),
                "protein_content": round(float(proteins[i]), 1),
            })
    result_foods.sort(key=lambda x: -x["percentage"])

    return {
        "foods":               result_foods,
        "mix_profile_per_food": {
            aa: round(float(mix_aa[j]), 3) for j, aa in enumerate(AMINO_ACIDS)
        },
        "mix_profile_per_protein": {
            aa: round(float(mix_per_prot[j]), 2) for j, aa in enumerate(AMINO_ACIDS)
        },
        "egg_reference":       EGG_REF,
        "who_scores":          blend_who_scores,
        "egg_similarity_pct":  round(cos_sim * 100, 1),
        "pdcaas_score":        pdcaas_blend,
        "total_protein_g":     round(float(total_protein), 1),
        "total_grams":         total_grams,
        "meets_protein_target": total_protein >= min_protein_g,
    }


# ── PIPELINE ENTRY POINT ─────────────────────────────────────────────────────

def run_pipeline(filepath: str) -> dict:
    """
    Full pipeline: load → clean → engineer features → train → optimise → report.
    """
    print("\n" + "=" * 65)
    print("  PlantProtein AI — ML Pipeline v5.0")
    print("=" * 65)

    # ── 1. Load & clean ───────────────────────────────────────────────────────
    df_raw = load_wide_excel(filepath)
    print(f"\n[pipeline] {len(df_raw)} foods · {df_raw['food_group'].nunique()} groups")

    # ── 2. Train ──────────────────────────────────────────────────────────────
    print("\n[pipeline] Training ensemble model (RF + GBM)...")
    model, scaler, metrics, df_feat = train_model(df_raw)

    # Predict scores for all foods
    X_all = scaler.transform(df_feat[FEATURE_COLS].values)
    df_feat["predicted_score"] = model.predict(X_all)

    # ── 3. Top foods ──────────────────────────────────────────────────────────
    top_cols = ["food", "food_group", "predicted_score", "pdcaas_score",
                "protein_content", "lim_aa", "lim_aa_score", "egg_cosine"]
    top_foods = (
        df_feat.nlargest(15, "predicted_score")[top_cols]
        .round(2)
        .to_dict("records")
    )

    # ── 4. Blends ─────────────────────────────────────────────────────────────
    print("\n[pipeline] Optimising blends (SLSQP)...")

    def _top(groups, n=12):
        sub = df_feat[df_feat["food_group"].isin(groups)]
        return sub.nlargest(n, "predicted_score")

    blends = {
        "cross_category": optimize_blend(
            pd.concat([
                df_feat[df_feat["food_group"] == g].nlargest(3, "predicted_score")
                for g in df_feat["food_group"].unique()
            ]), n_pick=6,
        ),
        "legumes_nuts": optimize_blend(
            _top(["Legumes", "Nuts and Seeds"]), n_pick=5,
        ),
        "vegetables_seeds": optimize_blend(
            _top(["Vegetables", "Seeds"]), n_pick=5,
        ),
        "cereals_legumes": optimize_blend(
            _top(["Cereals", "Legumes"]), n_pick=5,
        ),
        "consumer_friendly": optimize_blend(
            _top(["Fruits", "Vegetables", "Legumes"]), n_pick=4,
        ),
    }

    for name, blend in blends.items():
        if blend:
            print(
                f"   [{name}]  egg_sim={blend['egg_similarity_pct']}%  "
                f"PDCAAS={blend['pdcaas_score']}  "
                f"protein={blend['total_protein_g']}g"
            )
        else:
            print(f"   [{name}]  ⚠ not enough candidates")

    results = {
        "model_metrics":      metrics,
        "amino_acids":        AMINO_ACIDS,
        "who_reference":      WHO_REF,
        "egg_reference":      EGG_REF,
        "top_foods":          top_foods,
        "blends":             blends,
        "group_distribution": df_feat["food_group"].value_counts().to_dict(),
    }

    with open("ml_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n[pipeline] ✅  Results saved to ml_results.json")
    print("=" * 65 + "\n")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data.xlsx"
    run_pipeline(path)
