import { useEffect, useState } from "react";
import { api } from "../../api";
import { formatUsd } from "../../format";

const CANDIDATE_QUERIES = [
  "What deployment platform are we using now?",
  "What authentication method are we using?",
  "What database are we currently using?",
];

export default function IncidentReplay({ agentId, avgSavingsPerRun, refreshSignal }) {
  const [state, setState] = useState({ loading: true, incident: null, history: [] });
  const [fixed, setFixed] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function find() {
      setState({ loading: true, incident: null, history: [] });
      setFixed(false);
      for (const query of CANDIDATE_QUERIES) {
        try {
          const replay = await api.debugReplay(query, agentId);
          const baselineTop = replay.baseline.retrieved[0]?.memory;
          const activeSelected = replay.memtrace.selected.map((s) => s.memory).find((m) => m.status === "ACTIVE");
          if (baselineTop && activeSelected && baselineTop.id !== activeSelected.id) {
            const history = await api.getMemoryHistory(activeSelected.id);
            if (cancelled) return;
            setState({
              loading: false,
              incident: { query, baselineTop, activeSelected, answer: replay.memtrace.answer },
              history,
            });
            return;
          }
        } catch {
          // try the next candidate query
        }
      }
      if (!cancelled) setState({ loading: false, incident: null, history: [] });
    }

    find();
    return () => {
      cancelled = true;
    };
  }, [agentId, refreshSignal]);

  if (state.loading) return <div className="empty-note">Looking for a memory incident to replay…</div>;
  if (!state.incident) {
    return (
      <div className="empty-note">
        No stale-memory incident found yet. Let the agent learn two conflicting facts and it will show up here automatically.
      </div>
    );
  }

  const { baselineTop, answer } = state.incident;

  return (
    <div>
      <div style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 6 }}>Question</div>
      <div className="incident-quote" style={{ borderLeft: "3px solid var(--border)" }}>"{state.incident.query}"</div>

      <div style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 6 }}>Agent answer (naive search)</div>
      <div className="incident-quote">
        “The {baselineTop.subject} {baselineTop.predicate.replaceAll("_", " ")} {baselineTop.object}.”
      </div>
      <div className="incident-warning">⚠ OUTDATED MEMORY DETECTED</div>

      <div className="incident-timeline">
        {[...state.history].reverse().map((m, i, arr) => (
          <div key={m.id} style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div className="timeline-node">
              {m.object}
              <span className="date">
                {new Date(m.created_at).toLocaleDateString()} · {m.status}
              </span>
            </div>
            {i < arr.length - 1 && <span className="timeline-arrow">→ replaced by</span>}
          </div>
        ))}
      </div>

      <h4 style={{ marginBottom: 4 }}>Why it failed</h4>
      <p style={{ color: "var(--text-dim)", fontSize: 14, marginTop: 0 }}>
        The agent retrieved a historical memory ("{baselineTop.object}") instead of the current, active decision.
      </p>

      <div className="incident-cost">
        <div className="item">
          <div className="label">Estimated wasted cost</div>
          <div className="value">{formatUsd(avgSavingsPerRun || 0, { decimals: 4 })}</div>
        </div>
      </div>

      {!fixed ? (
        <button className="btn btn-primary" style={{ marginTop: 18 }} onClick={() => setFixed(true)}>
          Fix memory &amp; replay
        </button>
      ) : (
        <>
          <div className="kv-row" style={{ marginTop: 18 }}>
            <span className="k">Fix</span>
            <span style={{ color: "var(--green)", fontWeight: 700 }}>Memory already up to date</span>
          </div>
          <div style={{ fontSize: 12, color: "var(--text-dim)", margin: "14px 0 6px 0" }}>MEMTRACE answers</div>
          <div className="incident-quote fixed">“{answer}”</div>
        </>
      )}
    </div>
  );
}
