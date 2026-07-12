"""Feature-refresh consumer: event -> online store -> re-score -> alert (plan.md §14).

Implements the §14 flow verbatim: a simulated GST-filing / bank-transaction /
repayment event carries feature deltas for one loan; the consumer

  1. re-scores the loan on its CURRENT online features (pre-event grade),
  2. merges the deltas into the Feast-shaped online store,
  3. re-scores through the full pipeline (post-event grade),
  4. risk grade changed  -> insert an `alerts` row (dashboard alert),
     grade unchanged     -> nothing loud;
     either way the re-score lands in `risk_timeline` (silent trajectory).

The scoring function is injected (the serving app passes its own pipeline),
so this module stays free of model-loading side effects and is unit-testable
with a stub scorer.
"""

import logging
from typing import Callable

from src.serving import feature_store, store

logger = logging.getLogger("feature_refresh_consumer")

EVENT_TYPES = ("gst_filing", "bank_transaction", "repayment")
EVENTS_TOPIC = "loan_events"


def make_feature_refresh_handler(rescore: Callable[[int], dict]) -> Callable[[dict], dict]:
    """Build the bus handler around an injected `rescore(loan_id)` that returns
    {"calibrated_pd_12m": float, "risk_grade": str} from the full pipeline."""

    def handle(event: dict) -> dict:
        loan_id = int(event["loan_id"])
        event_type = event.get("type")
        updates = event.get("feature_updates") or {}

        pre = rescore(loan_id)
        if updates:
            feature_store.update_features(loan_id, updates)
        post = rescore(loan_id)

        store.insert_timeline(loan_id, post["calibrated_pd_12m"], post["risk_grade"])

        changed = post["risk_grade"] != pre["risk_grade"]
        if changed:
            message = (
                f"{event_type or 'event'} moved loan {loan_id} from grade "
                f"{pre['risk_grade']} to {post['risk_grade']} "
                f"(PD {pre['calibrated_pd_12m']:.4f} -> {post['calibrated_pd_12m']:.4f})"
            )
            store.insert_alert(
                loan_id, event_type, pre["risk_grade"], post["risk_grade"], message
            )
            logger.info("ALERT %s", message)
        else:
            logger.info(
                "silent timeline update loan=%s grade=%s pd=%.4f (event %s)",
                loan_id,
                post["risk_grade"],
                post["calibrated_pd_12m"],
                event.get("event_id"),
            )
        return {
            "loan_id": loan_id,
            "event_id": event.get("event_id"),
            "grade_changed": changed,
            "pre": pre,
            "post": post,
        }

    return handle
