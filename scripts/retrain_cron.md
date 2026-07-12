# Scheduled retrain check (plan.md §13.2, §13.4)

`scripts/retrain_check.py` runs the monthly drift -> retrain loop: Evidently/PSI
drift report on the latest window; if PSI > 0.2 on any Tier-1 feature it trains
a challenger and gates it against the current Production model on the OOT test
window (never auto-promoting a worse model). Out-of-cycle runs can also be
fired manually or by the streaming pipeline.

## Monthly schedule — 02:00 on the 1st of every month

### Windows (Task Scheduler)

```powershell
schtasks /Create /TN "NexusMSME-RetrainCheck" /SC MONTHLY /D 1 /ST 02:00 `
  /TR "cmd /c cd /d C:\Users\raned\OneDrive\Desktop\Nexus-MSME-EWS && uv run python scripts\retrain_check.py >> reports\retrain_check.log 2>&1"
```

### Linux / macOS (cron)

```cron
# m h dom mon dow  command
0 2 1 * * cd /opt/nexus-msme-ews && uv run python scripts/retrain_check.py >> reports/retrain_check.log 2>&1
```

Exit codes: `0` = no drift, or challenger promoted; `1` = drift fired but the
challenger was HELD by the promotion gate — flag for manual review (§13.2
flowchart "Hold" branch), so wire the non-zero exit to your alerting.

## Upgrade path (§13.4)

- **Prefect** (`prefect deploy` with a cron schedule) is the next step for a
  slightly more production-realistic look: retries, run history UI, and the
  drift-check / retrain / gate steps as separate observable tasks.
- **Apache Airflow** is the named production orchestration choice once this
  moves beyond a single-node demo (DAG = feature refresh -> re-score -> drift
  report -> conditional retrain -> gated promotion).
