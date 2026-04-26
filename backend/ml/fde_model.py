"""
FDE Personal ML Model — MedLogic
=================================
ALGORITHM: Weighted Component Intersection Scoring (WCIS)
  — a custom rule-based Bayesian-inspired incremental learning algorithm.

WHY NOT A STANDARD ML MODEL?
  Standard classifiers (RandomForest, SVM etc.) need labelled training data
  per patient, which doesn't exist at first visit. FDE is rare and patient-
  specific. WCIS learns incrementally from each FDE event — like online
  Bayesian updating — and is fully personalized.

SCORING LOGIC:
  1. First FDE  → P(component=culprit | 1 event) — all components get +2.0
  2. Repeat FDE → intersection with ALL past FDE component sets is computed.
     Components appearing in N past events: weight = BASE + N*REPEAT_BONUS
     This is analogous to a Naive-Bayes likelihood update per component.
  3. Risk(drug) = Σ score(c) for c in drug_components  (additive risk model)
  4. Alternatives: scan DB for drugs with same Use and zero flagged components.

COMPLEXITY: O(E * C) per update where E=FDE events, C=components per drug.
"""

from database import get_db

# ── Scoring constants ──
FIRST_WEIGHT   = 2.0   # weight on first FDE for each component
REPEAT_BASE    = 3.0   # base weight when component recurs in a 2nd+ FDE
REPEAT_BONUS   = 1.5   # extra weight per additional past FDE event it appeared in
NON_OVERLAP_W  = 0.5   # small weight for new components on a repeat FDE (could be culprit)

THRESH_LOW  = 3.0
THRESH_HIGH = 7.0


# ─────────────────────────────────────────────
def get_component_scores(patient_id: str) -> dict:
    conn = get_db()
    rows = conn.execute(
        "SELECT component, score, fde_count FROM component_scores WHERE patient_id=?",
        (patient_id,)
    ).fetchall()
    conn.close()
    return {r["component"]: {"score": r["score"], "fde_count": r["fde_count"]} for r in rows}


def _upsert_score(conn, patient_id, component, delta):
    exists = conn.execute(
        "SELECT score FROM component_scores WHERE patient_id=? AND component=?",
        (patient_id, component)
    ).fetchone()
    if exists:
        conn.execute(
            "UPDATE component_scores SET score=score+?, fde_count=fde_count+1, last_updated=CURRENT_TIMESTAMP WHERE patient_id=? AND component=?",
            (delta, patient_id, component)
        )
    else:
        conn.execute(
            "INSERT INTO component_scores (patient_id, component, score, fde_count) VALUES (?,?,?,1)",
            (patient_id, component, delta)
        )


def get_past_fde_components(patient_id: str, exclude_id: int = None) -> list:
    """Return list of component-sets from all past FDE events for this patient."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, medicine_name, components FROM dosage_history WHERE patient_id=? AND fde_occurred=1",
        (patient_id,)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        if exclude_id and r["id"] == exclude_id:
            continue
        result.append({
            "id": r["id"],
            "medicine": r["medicine_name"],
            "components": set(c.strip() for c in r["components"].split(","))
        })
    return result


# ─────────────────────────────────────────────
def process_fde_event(patient_id: str, new_components: list, dosage_id: int) -> dict:
    """
    Core WCIS update step called when patient reports an FDE.
    Updates component scores in DB and returns analysis dict.
    """
    new_set = set(c.strip().lower() for c in new_components)
    past    = get_past_fde_components(patient_id, exclude_id=dosage_id)

    analysis = {
        "algorithm": "WCIS — Weighted Component Intersection Scoring",
        "is_first_fde":        len(past) == 0,
        "past_fde_count":      len(past),
        "new_components":      list(new_set),
        "common_components":   [],
        "culprit_components":  [],
        "weights_applied":     {},
        "explanation":         ""
    }

    conn = get_db()

    if len(past) == 0:
        # ── FIRST FDE: flag everything ──
        for comp in new_set:
            _upsert_score(conn, patient_id, comp, FIRST_WEIGHT)
            analysis["weights_applied"][comp] = FIRST_WEIGHT
        analysis["culprit_components"] = sorted(new_set)
        analysis["explanation"] = (
            f"First FDE recorded. All {len(new_set)} components flagged with "
            f"base weight {FIRST_WEIGHT}. System will narrow down culprits on repeat events."
        )

    else:
        # ── REPEAT FDE: WCIS intersection update ──
        overlap_count = {}   # component → number of past FDE events it appeared in
        for p in past:
            for comp in new_set & p["components"]:
                overlap_count[comp] = overlap_count.get(comp, 0) + 1

        for comp in new_set:
            n = overlap_count.get(comp, 0)
            if n > 0:
                weight = REPEAT_BASE + (n - 1) * REPEAT_BONUS
                analysis["common_components"].append(comp)
            else:
                weight = NON_OVERLAP_W
            _upsert_score(conn, patient_id, comp, weight)
            analysis["weights_applied"][comp] = round(weight, 2)

        # Rank culprits by overlap count then weight
        analysis["culprit_components"] = sorted(
            overlap_count.keys(), key=lambda c: overlap_count[c], reverse=True
        )
        analysis["explanation"] = (
            f"Repeat FDE #{len(past)+1}. Intersection with {len(past)} past FDE events computed. "
            f"{len(analysis['common_components'])} common components found and up-weighted "
            f"(WCIS weight = {REPEAT_BASE} + overlaps×{REPEAT_BONUS})."
        )

    conn.commit()
    conn.close()

    # Attach current scores for display
    scores = get_component_scores(patient_id)
    analysis["current_scores"] = {
        c: round(scores.get(c, {}).get("score", 0), 2) for c in new_set
    }
    return analysis


# ─────────────────────────────────────────────
def calculate_risk(patient_id: str, components: list) -> dict:
    """
    Risk score = Σ component_score(c) for each c in drug's components.
    Normalized slightly for large component counts.
    """
    comp_list = [c.strip().lower() for c in components]
    scores    = get_component_scores(patient_id)

    comp_risks  = {c: round(scores.get(c, {}).get("score", 0), 2) for c in comp_list}
    total       = sum(comp_risks.values())
    normalized  = round(total / (1 + 0.08 * max(len(comp_list) - 3, 0)), 2)

    if normalized >= THRESH_HIGH:
        level, color = "HIGH",     "#ef4444"
    elif normalized >= THRESH_LOW:
        level, color = "MODERATE", "#f59e0b"
    else:
        level, color = "LOW",      "#22c55e"

    flagged = [c for c, s in comp_risks.items() if s >= THRESH_LOW]

    return {
        "risk_score":         normalized,
        "raw_score":          round(total, 2),
        "risk_level":         level,
        "risk_color":         color,
        "component_risks":    comp_risks,
        "flagged_components": flagged,
        "algorithm_note":     f"WCIS additive score: Σ weights({len(comp_list)} components) = {total:.2f}, normalized = {normalized:.2f}"
    }


# ─────────────────────────────────────────────
def find_safe_alternatives(patient_id: str, target_use: str, df, col_n, col_c, col_u) -> list:
    scores   = get_component_scores(patient_id)
    risky    = {c for c, v in scores.items() if v["score"] >= THRESH_LOW}
    safe     = []

    for _, row in df.iterrows():
        comps   = row[col_c]
        use_val = str(row.get(col_u, "")).strip()

        if target_use and target_use.lower() not in use_val.lower():
            continue

        if not set(comps) & risky:
            safe.append({
                "name":       row[col_n],
                "components": comps[:6],
                "used_for":   use_val,
                "risk_level": str(row.get("Risk_Level", "")).strip()
            })
        if len(safe) >= 6:
            break
    return safe
