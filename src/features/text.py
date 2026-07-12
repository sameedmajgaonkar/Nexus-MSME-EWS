"""Phase 5 text pipeline: synthetic officer notes -> MiniLM embeddings -> FinBERT
sentiment -> PCA(12) dense vectors + regex distress flag (plan.md §8.3, §9.5).

Honesty note (plan.md §6.5): notes are SYNTHETIC, drawn from a finite template
pool; template tone conditions ONLY on observable behavioral columns
(late_installment_rate, bureau_overdue_flag), NEVER on TARGET. Every row
carries data_provenance='synthetic_text'.

Performance rule: the pool has ~72 unique sentences, so embeddings and FinBERT
run on the UNIQUE sentences only and are mapped back to the 307K loans —
never 307K forward passes.

Heavy imports (sentence-transformers, transformers/torch) are deliberately
inside functions so unit tests can exercise the non-network parts without
loading or downloading any model.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

PROVENANCE = "synthetic_text"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # plan.md §8.3 step 1
FINBERT_MODEL = "ProsusAI/finbert"  # plan.md §8.3 step 2
N_PCA_COMPONENTS = 12  # plan.md §8.3 step 3 (10-20 components, dense — §9.5)

# plan.md §8.3 step 4 — distress keywords; substring match also catches
# "delayed", "delays", "disputes", "closures".
DISTRESS_PATTERN = re.compile(r"delay|disput|stoppage|closure", re.IGNORECASE)

TONES = ("positive", "neutral", "distress")

POSITIVE_TEMPLATES = [
    "Unit visit completed; production lines running at full capacity and order book healthy.",
    "Promoter reports strong festive-season sales and timely receivables collection.",
    "Repayments regular for the past year; account conduct satisfactory.",
    "GST filings on time and turnover trending upward quarter on quarter.",
    "New purchase order from an anchor buyer expected to lift monthly revenue.",
    "Working capital utilisation comfortable; no overdrawals observed.",
    "Borrower expanded machinery capacity funded largely from internal accruals.",
    "Cash flows steady; promoter maintains healthy balance in the current account.",
    "Stock and receivables audit found records well maintained and margins stable.",
    "Supplier payments current; trade references speak favourably of the firm.",
    "The firm added two new customers this quarter, reducing dependence on a single buyer.",
    "Interest serviced on due date every month; no follow-up required.",
    "Site inspection satisfactory; inventory levels consistent with declared turnover.",
    "Promoter injected additional equity to support the expansion plan.",
    "Export order pipeline strong; advance payments received from the overseas buyer.",
    "Account shows healthy transaction velocity with consistent credits.",
    "Firm won a rate contract with a public-sector buyer, improving revenue visibility.",
    "Margins improved after renegotiating raw material contracts with vendors.",
    "All statutory dues paid on schedule; compliance record clean.",
    "Collections efficiency improved; debtor days reduced noticeably this quarter.",
    "The borrower prepaid one installment ahead of schedule from surplus cash.",
    "Order flow from the anchor OEM remains steady and growing.",
    "Audited financials show improving net worth and comfortable gearing.",
    "Business is seasonal but well managed; promoter keeps adequate liquidity buffers.",
]

NEUTRAL_TEMPLATES = [
    "Routine annual review completed; no significant change in business profile.",
    "The firm operates a mid-sized fabrication workshop in the industrial estate.",
    "Documents received for renewal; financial statements under review.",
    "Borrower requested a statement of account for reconciliation purposes.",
    "Stock statement submitted for the month; figures consistent with prior submissions.",
    "Unit engaged in trading of electrical components; ownership unchanged.",
    "Insurance policy for hypothecated stock renewed for the current year.",
    "Promoter attended the branch to update KYC records.",
    "Turnover broadly flat compared with the previous financial year.",
    "The account was migrated to the new core banking platform this month.",
    "Firm employs about twenty workers across two shifts.",
    "Renewal proposal under process; awaiting updated GST returns.",
    "Borrower operates from rented premises with a long-term lease.",
    "No inspection was due this quarter; next visit planned as per calendar.",
    "The company supplies machined parts to several regional assemblers.",
    "Limits utilised within sanctioned levels during the review period.",
    "Change of authorised signatory recorded as per board resolution.",
    "Business mix unchanged; roughly sixty percent institutional and forty percent retail.",
    "Borrower enquired about a top-up facility; details shared for evaluation.",
    "Quarterly stock inspection planned with the empanelled agency.",
    "The firm maintains its current account and cash credit with our branch.",
    "Promoter family has been in the same line of business for two decades.",
    "Standard covenants reviewed; no waivers sought during the period.",
    "Financial statements awaited; provisional figures indicate stable operations.",
]

DISTRESS_TEMPLATES = [
    "Payments from the anchor buyer are facing significant delay, straining cash flows.",
    "Production stoppage reported at the unit due to shortage of raw material.",
    "Promoter cited a GST dispute that has frozen input tax credit this quarter.",
    "The firm's largest customer is delaying settlements beyond ninety days.",
    "Partial closure of one production line observed during the site visit.",
    "Wage payments delayed last month; worker unrest reported at the factory.",
    "A legal dispute with the key supplier has interrupted raw material supply.",
    "Borrower warned of possible closure of the retail outlet if sales do not recover.",
    "EMI paid after repeated follow-up; borrower attributed the delay to stuck receivables.",
    "Power supply stoppage for unpaid dues halted operations for several days.",
    "Receivables under dispute with two counterparties; recovery timeline uncertain.",
    "Order flow from the anchor OEM has thinned; unit reports intermittent stoppage of work.",
    "Interest servicing delayed twice this quarter; account slipping toward SMA status.",
    "Buyer disputes over quality claims have led to withheld payments.",
    "Temporary closure of the workshop following a fire safety notice.",
    "Salary delays and vendor dues mounting; promoter seeking emergency funding.",
    "Cheque issued to a key vendor returned; supplier threatening supply stoppage.",
    "GST filings delayed for two consecutive months; turnover decline suspected.",
    "The unit reported closure of its second shift owing to weak demand.",
    "Promoter locked in a partnership dispute; day-to-day management affected.",
    "Delivery delays have triggered penalty clauses with the principal buyer.",
    "Stock audit found slow-moving inventory; unit facing intermittent stoppage of production.",
    "Anchor buyer invoked a payment dispute, freezing the firm's biggest receivable.",
    "Borrower requested moratorium citing prolonged delay in government subsidy release.",
]

TEMPLATE_POOLS = {
    "positive": POSITIVE_TEMPLATES,
    "neutral": NEUTRAL_TEMPLATES,
    "distress": DISTRESS_TEMPLATES,
}


def distress_keyword_flag(notes: pd.Series) -> pd.Series:
    """Binary regex flag for 'delay', 'dispute', 'stoppage', 'closure' (plan.md §8.3)."""
    return notes.str.contains(DISTRESS_PATTERN).astype(int).rename("distress_keyword_flag")


def _tone_probabilities(late: np.ndarray, overdue: np.ndarray) -> np.ndarray:
    """Row-wise (positive, neutral, distress) probabilities from observable behavior only."""
    distress = np.clip(0.04 + 1.3 * late + 0.35 * overdue, 0.0, 0.85)
    positive = np.clip(0.55 - 1.6 * late - 0.30 * overdue, 0.05, 1.0)
    neutral = np.clip(1.0 - distress - positive, 0.05, None)
    probs = np.stack([positive, neutral, distress], axis=1)
    return probs / probs.sum(axis=1, keepdims=True)


def generate_officer_notes(features_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Synthetic loan-officer notes from the finite template pool, one per SK_ID_CURR.

    Tone (positive/neutral/distress) conditions ONLY on observable behavior
    (late_installment_rate, bureau_overdue_flag) — NEVER on TARGET. This is a
    documented simulation (plan.md §6.5); provenance 'synthetic_text'.
    """
    rng = np.random.default_rng(seed)
    late = features_df["late_installment_rate"].fillna(0).to_numpy(dtype=float)
    overdue = features_df["bureau_overdue_flag"].fillna(0).to_numpy(dtype=float)
    n = len(features_df)

    probs = _tone_probabilities(late, overdue)
    tone_idx = (rng.random(n)[:, None] > probs.cumsum(axis=1)).sum(axis=1)
    template_draw = rng.integers(0, 1_000_000, n)

    notes = np.empty(n, dtype=object)
    tones = np.empty(n, dtype=object)
    for idx, tone in enumerate(TONES):
        pool = np.asarray(TEMPLATE_POOLS[tone], dtype=object)
        mask = tone_idx == idx
        notes[mask] = pool[template_draw[mask] % len(pool)]
        tones[mask] = tone

    return pd.DataFrame(
        {
            "SK_ID_CURR": features_df["SK_ID_CURR"].to_numpy(),
            "officer_note": notes,
            "note_tone": tones,
            "data_provenance": PROVENANCE,
        }
    )


def unique_note_index(notes: pd.Series) -> tuple[list[str], np.ndarray]:
    """Deduplicate notes: returns (unique_sentences, codes) with uniques[codes[i]] == notes[i].

    This is the mapping that keeps the 307K-loan pipeline down to ~72 model
    forward passes (PHASES.md Phase 5 performance rule).
    """
    cat = pd.Categorical(notes)
    return list(cat.categories), cat.codes.astype(int)


def embed_notes(unique_sentences: list[str], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """MiniLM sentence embeddings (384-dim) for the UNIQUE sentences only."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return np.asarray(
        model.encode(list(unique_sentences), show_progress_bar=False, convert_to_numpy=True)
    )


def finbert_sentiment(unique_sentences: list[str], model_name: str = FINBERT_MODEL) -> pd.DataFrame:
    """FinBERT positive/negative/neutral sentiment for the UNIQUE sentences only."""
    from transformers import pipeline

    try:
        classifier = pipeline("sentiment-analysis", model=model_name)
    except (KeyError, ValueError):  # transformers v5 task-alias drift
        classifier = pipeline("text-classification", model=model_name)
    outputs = classifier(list(unique_sentences), truncation=True)
    return pd.DataFrame(
        {
            "sentence": list(unique_sentences),
            "sentiment_label": [o["label"].lower() for o in outputs],
            "sentiment_score": [float(o["score"]) for o in outputs],
        }
    )


def signed_sentiment(labels: pd.Series, scores: pd.Series) -> pd.Series:
    """Numeric encoding for the model: +score / -score / 0 for pos/neg/neutral."""
    sign = labels.map({"positive": 1.0, "negative": -1.0, "neutral": 0.0}).fillna(0.0)
    return (sign * scores).rename("sentiment_signed")


def build_text_features(
    notes_df: pd.DataFrame, n_components: int = N_PCA_COMPONENTS, seed: int = 42
) -> pd.DataFrame:
    """Full §8.3 pipeline keyed by SK_ID_CURR: text_pc_1..12 (dense vector, not a
    scalar — plan.md §9.5), sentiment label/score/signed, distress_keyword_flag.

    Embeds and classifies the unique template sentences only, then maps back.
    Downloads MiniLM + FinBERT from HuggingFace on first use.
    """
    from sklearn.decomposition import PCA

    uniques, codes = unique_note_index(notes_df["officer_note"])

    embeddings = embed_notes(uniques)
    n_comp = min(n_components, len(uniques) - 1, embeddings.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    components_unique = pca.fit_transform(embeddings)
    components = components_unique[codes]

    sentiment_unique = finbert_sentiment(uniques)
    labels = sentiment_unique["sentiment_label"].to_numpy(dtype=object)[codes]
    scores = sentiment_unique["sentiment_score"].to_numpy()[codes]

    flags_unique = distress_keyword_flag(pd.Series(uniques)).to_numpy()

    out = pd.DataFrame(
        components, columns=[f"text_pc_{i + 1}" for i in range(n_comp)]
    )
    out.insert(0, "SK_ID_CURR", notes_df["SK_ID_CURR"].to_numpy())
    out["sentiment_label"] = labels
    out["sentiment_score"] = scores
    out["sentiment_signed"] = signed_sentiment(out["sentiment_label"], out["sentiment_score"])
    out["distress_keyword_flag"] = flags_unique[codes]
    out["data_provenance"] = PROVENANCE
    return out
