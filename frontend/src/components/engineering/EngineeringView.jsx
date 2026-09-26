import { useState } from "react";
import { api, DEFAULT_AGENT_ID, DEFAULT_CONVERSATION_ID } from "../../api";

export default function EngineeringView() {
  const [query, setQuery] = useState("What database are we currently using?");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function run() {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const r = await api.debugQuery(query.trim(), DEFAULT_AGENT_ID, DEFAULT_CONVERSATION_ID);
      setResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="section card">
        <h3 className="card-title">Run a query through the pipeline</h3>
        <div className="ask-bar">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && run()}
          />
          <button className="btn btn-primary" onClick={run} disabled={loading}>
            {loading ? "Running…" : "Run debug query"}
          </button>
        </div>
        {error && <p style={{ color: "var(--red)", fontSize: 13 }}>{error}</p>}
      </div>

      {result && (
        <>
          <div className="section card">
            <h3 className="card-title">Answer</h3>
            <p style={{ fontSize: 15 }}>{result.answer}</p>
            <div className="kv-row">
              <span className="k">LangGraph pipeline</span>
              <span className="mono" style={{ fontSize: 12 }}>
                ANALYZE_QUERY → RETRIEVE_MEMORY → TRAVERSE_GRAPH → FILTER_TEMPORAL_MEMORY → BUILD_CONTEXT →
                GENERATE_RESPONSE
              </span>
            </div>
            <div className="kv-row">
              <span className="k">LangSmith run ID</span>
              <span className="mono">{result.langsmith_run_id ?? "tracing disabled (no LANGCHAIN_API_KEY configured)"}</span>
            </div>
            <div className="kv-row">
              <span className="k">Candidates retrieved / selected / excluded</span>
              <span>
                {result.retrieved.length} / {result.context.selected.length} / {result.excluded.length}
              </span>
            </div>
            <div className="kv-row">
              <span className="k">Context token estimate</span>
              <span>{result.context.token_estimate}</span>
            </div>
          </div>

          <div className="grid grid-2 section">
            <div className="card">
              <h3 className="card-title">Selected context (why used)</h3>
              {result.context.selected.length === 0 && <div className="empty-note">Nothing selected.</div>}
              {result.context.selected.map((sm) => (
                <div className="eng-mem-card" key={sm.memory.id}>
                  <span className={`badge ${sm.memory.status}`}>{sm.memory.status}</span>
                  <span className="triple">
                    {sm.memory.subject} {sm.memory.predicate} {sm.memory.object}
                  </span>
                  <div className="content">{sm.memory.content}</div>
                  <div className="reason used">
                    score {sm.score.toFixed(2)} · {sm.retrieval_reason.join(", ")}
                  </div>
                </div>
              ))}
            </div>
            <div className="card">
              <h3 className="card-title">Excluded (why ignored)</h3>
              {result.excluded.length === 0 && <div className="empty-note">Nothing excluded.</div>}
              {result.excluded.map((em) => (
                <div className="eng-mem-card" key={em.memory.id}>
                  <span className={`badge ${em.memory.status}`}>{em.memory.status}</span>
                  <span className="triple">
                    {em.memory.subject} {em.memory.predicate} {em.memory.object}
                  </span>
                  <div className="reason ignored">{em.reason}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="section card">
            <h3 className="card-title">Raw context sent to the model</h3>
            <pre style={{ whiteSpace: "pre-wrap", fontSize: 12, color: "var(--text-dim)", margin: 0 }}>
              {result.context.context_text}
            </pre>
          </div>
        </>
      )}
    </div>
  );
}
