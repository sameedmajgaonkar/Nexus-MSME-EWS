import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchAlerts,
  fetchDriftReport,
  fetchLoans,
  fetchMetrics,
  fetchOverrides,
  fetchPortfolioSummary,
  fetchTimeline,
  scoreLoan,
  submitDecision,
} from "./api.js";
import BorrowerView from "./components/BorrowerView.jsx";
import DriverChart from "./components/DriverChart.jsx";
import EventSimulator from "./components/EventSimulator.jsx";
import FairnessAudit from "./components/FairnessAudit.jsx";
import HazardChart from "./components/HazardChart.jsx";
import LoanList from "./components/LoanList.jsx";
import MetricsTable from "./components/MetricsTable.jsx";
import OverridePanel from "./components/OverridePanel.jsx";
import PortfolioSummary from "./components/PortfolioSummary.jsx";
import RbiMapping from "./components/RbiMapping.jsx";
import RiskTimeline from "./components/RiskTimeline.jsx";
import ScoreCard from "./components/ScoreCard.jsx";
import StressTest from "./components/StressTest.jsx";

const TABS = ["Console", "Stress Test", "Fairness", "RBI Mapping", "Borrower View"];
const ALERT_POLL_MS = 4000;

export default function App() {
  const [tab, setTab] = useState("Console");
  const [loans, setLoans] = useState([]);
  const [score, setScore] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [overrides, setOverrides] = useState([]);
  const [summary, setSummary] = useState(null);
  const [drift, setDrift] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [alertBanner, setAlertBanner] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const lastAlertId = useRef(null); // null until the first poll seeds it
  const scoreRef = useRef(null);
  scoreRef.current = score;

  useEffect(() => {
    fetchLoans().then(setLoans).catch((e) => setError(e.message));
    fetchMetrics().then(setMetrics).catch(() => {});
    fetchOverrides().then(setOverrides).catch(() => {});
    fetchPortfolioSummary().then(setSummary).catch(() => {});
    fetchDriftReport().then(setDrift).catch(() => {});
  }, []);

  const refreshTimeline = useCallback((loanId) => {
    if (loanId == null) return;
    fetchTimeline(loanId).then(setTimeline).catch(() => {});
  }, []);

  // Live alerts: poll every ~4s; a NEW alert raises the banner and refreshes
  // the open loan's score + trajectory (plan.md §14 / Phase 11 front half).
  useEffect(() => {
    async function poll() {
      try {
        const alerts = await fetchAlerts(10);
        const maxId = alerts.length ? Math.max(...alerts.map((a) => a.id)) : 0;
        if (lastAlertId.current === null) {
          lastAlertId.current = maxId; // seed: don't replay history on load
          return;
        }
        const fresh = alerts.filter((a) => a.id > lastAlertId.current);
        if (fresh.length === 0) return;
        lastAlertId.current = maxId;
        setAlertBanner(fresh[0]); // newest first
        const open = scoreRef.current;
        if (open && fresh.some((a) => a.loan_id === open.loan_id)) {
          scoreLoan(open.loan_id).then(setScore).catch(() => {});
          fetchTimeline(open.loan_id).then(setTimeline).catch(() => {});
        }
      } catch {
        /* backend briefly unavailable — keep polling */
      }
    }
    poll(); // seed immediately at mount so an alert landing within the first
    // poll interval is not mistaken for history
    const timer = setInterval(poll, ALERT_POLL_MS);
    return () => clearInterval(timer);
  }, []);

  async function handleSelect(loanId) {
    setLoading(true);
    setError(null);
    try {
      setScore(await scoreLoan(loanId));
      refreshTimeline(loanId);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleDecision(decision, reason, delta) {
    if (!score) return;
    await submitDecision(score.loan_id, decision, reason, delta);
    setOverrides(await fetchOverrides());
  }

  function handleEventPublished() {
    // The consumer re-scores within a few seconds; the alert poller catches
    // grade moves — this just refreshes the silent trajectory shortly after.
    const loanId = score?.loan_id;
    setTimeout(() => refreshTimeline(loanId), 3000);
  }

  return (
    <div className="app">
      <header className="banner">
        <div>
          <h1>MSME RiskPulse</h1>
          <span className="subtitle">Risk Officer Console — MVP</span>
        </div>
        <nav className="tab-bar">
          {TABS.map((t) => (
            <button
              key={t}
              className={`tab-btn ${tab === t ? "active" : ""}`}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          ))}
        </nav>
        <div className="disclosure">
          ⚠ AI-assisted assessment. A human officer holds final decision authority.
        </div>
      </header>

      {alertBanner && (
        <div className="alert-banner">
          <strong>⚡ Live alert:</strong> loan #{alertBanner.loan_id} moved grade{" "}
          <strong>
            {alertBanner.old_grade} → {alertBanner.new_grade}
          </strong>{" "}
          after a {alertBanner.event_type?.replace(/_/g, " ")} event. {alertBanner.message}
          <button className="alert-dismiss" onClick={() => setAlertBanner(null)}>
            ✕
          </button>
        </div>
      )}

      {tab === "Console" && (
        <div className="layout">
          <aside className="sidebar">
            <PortfolioSummary summary={summary} drift={drift} />
            <LoanList loans={loans} selectedId={score?.loan_id} onSelect={handleSelect} />
          </aside>

          <main className="main">
            {error && <div className="error">{error}</div>}
            {loading && (
              <div className="loading">
                Scoring… established loans return in under a second; new-to-credit loans route
                to the TabPFN thin-file specialist (~30s on CPU).
              </div>
            )}
            {!score && !loading && (
              <div className="placeholder">Select a loan to run the scoring pipeline.</div>
            )}
            {score && !loading && (
              <>
                <ScoreCard score={score} />
                <div className="charts">
                  <HazardChart curve={score.hazard_curve} />
                  <DriverChart drivers={score.top_drivers} />
                </div>
                <RiskTimeline timeline={timeline} />
                <EventSimulator loanId={score.loan_id} onPublished={handleEventPublished} />
                <OverridePanel
                  key={score.loan_id}
                  onDecision={handleDecision}
                  overrides={overrides}
                />
              </>
            )}
            {metrics && <MetricsTable metrics={metrics} />}
          </main>
        </div>
      )}

      {tab === "Stress Test" && (
        <div className="layout single">
          <StressTest />
        </div>
      )}
      {tab === "Fairness" && (
        <div className="layout single">
          <FairnessAudit />
        </div>
      )}
      {tab === "RBI Mapping" && (
        <div className="layout single">
          <RbiMapping />
        </div>
      )}
      {tab === "Borrower View" && (
        <div className="layout single">
          <BorrowerView loans={loans} />
        </div>
      )}
    </div>
  );
}
