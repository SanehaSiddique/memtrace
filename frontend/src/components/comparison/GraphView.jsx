import { useEffect, useState, useRef } from "react";
import { api } from "../../api";

/**
 * Neo4j Graph View Visualization for Agent 2 (docs/IMPLEMENTATION.md §9).
 * Renders nodes (facts) and edges (SUPERSEDED_BY, etc.).
 * Active facts render bright/green; stale facts render visually distinct (greyed out / strikethrough),
 * showing how memory updates silently in the background without context pollution.
 */
export default function GraphView({ sessionId, refreshSignal }) {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    api
      .getGraphSnapshot(sessionId || "default_session")
      .then((data) => {
        if (!cancelled && data) {
          setGraphData({
            nodes: data.nodes || [],
            edges: data.edges || [],
          });
        }
      })
      .catch((err) => {
        console.warn("Failed to fetch graph snapshot:", err);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [sessionId, refreshSignal]);

  const activeNodes = graphData.nodes.filter((n) => n.status !== "STALE" && n.status !== "REPLACED");
  const staleNodes = graphData.nodes.filter((n) => n.status === "STALE" || n.status === "REPLACED");

  return (
    <div className="card graph-view-card">
      <div className="section-heading" style={{ marginBottom: 12 }}>
        <div>
          <h3 className="card-title" style={{ margin: 0 }}>Neo4j Fact Knowledge Graph (Agent 2 Memory)</h3>
          <p className="card-note" style={{ margin: "4px 0 0 0" }}>
            Visualizes dynamic fact extraction and JEV-powered <code>SUPERSEDED_BY</code> edges. Stale facts are excluded from LLM reasoning turns.
          </p>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span className="badge badge-active">{activeNodes.length} Active</span>
          <span className="badge" style={{ background: "rgba(255,107,107,0.15)", color: "var(--red)" }}>
            {staleNodes.length} Stale / Superseded
          </span>
        </div>
      </div>

      {loading && graphData.nodes.length === 0 ? (
        <div className="empty-note">Loading Neo4j graph data...</div>
      ) : graphData.nodes.length === 0 ? (
        <div className="empty-note">
          No facts stored yet. Run a prompt that mentions a fact (e.g. "We are migrating to PostgreSQL next week") to watch Agent 2 extract facts and form graph relationships.
        </div>
      ) : (
        <div className="graph-container-flex" style={{ display: "flex", gap: 16, minHeight: 340 }}>
          {/* Visual Node List / Graph Canvas */}
          <div className="graph-canvas-box" style={{ flex: 1, background: "var(--panel-2)", borderRadius: "var(--radius-sm)", padding: 16, border: "1px solid var(--border)", overflowY: "auto", maxHeight: 480 }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text-dim)", textTransform: "uppercase", marginBottom: 10 }}>
              Fact Nodes & Lineage
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {graphData.nodes.map((node) => {
                const isStale = node.status === "STALE" || node.status === "REPLACED";
                const isSelected = selectedNode?.id === node.id;
                const outgoingEdges = graphData.edges.filter((e) => e.source === node.id);

                return (
                  <div
                    key={node.id}
                    onClick={() => setSelectedNode(node)}
                    style={{
                      background: isStale ? "rgba(38, 44, 56, 0.4)" : "var(--panel)",
                      border: isSelected ? "2px solid var(--green)" : isStale ? "1px dashed var(--border)" : "1px solid var(--border)",
                      borderRadius: 8,
                      padding: "10px 14px",
                      cursor: "pointer",
                      opacity: isStale ? 0.6 : 1.0,
                      transition: "all 0.15s ease",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
                      <span style={{ fontSize: 11, fontFamily: "monospace", color: "var(--text-faint)" }}>
                        {node.id}
                      </span>
                      <span
                        className={`badge ${!isStale ? "badge-active" : ""}`}
                        style={isStale ? { background: "rgba(255,107,107,0.15)", color: "var(--red)", fontSize: 10 } : { fontSize: 10 }}
                      >
                        {node.status}
                      </span>
                    </div>

                    <div
                      style={{
                        fontSize: 14,
                        color: isStale ? "var(--text-faint)" : "var(--text)",
                        textDecoration: isStale ? "line-through" : "none",
                        lineHeight: 1.4,
                      }}
                    >
                      {node.text}
                    </div>

                    {outgoingEdges.length > 0 && (
                      <div style={{ marginTop: 8, paddingTop: 6, borderTop: "1px solid var(--border)", fontSize: 12, color: "var(--amber)" }}>
                        {outgoingEdges.map((e, idx) => (
                          <div key={idx} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                            <span>↳ <strong>{e.relation}</strong></span>
                            <span style={{ fontFamily: "monospace", color: "var(--text-dim)" }}>→ {e.target}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Details Sidebar */}
          <div style={{ width: 280, background: "var(--panel)", borderRadius: "var(--radius-sm)", padding: 16, border: "1px solid var(--border)" }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text-dim)", textTransform: "uppercase", marginBottom: 12 }}>
              Fact Inspector
            </div>

            {selectedNode ? (
              <div>
                <div style={{ marginBottom: 10 }}>
                  <span style={{ fontSize: 11, color: "var(--text-dim)" }}>Node ID</span>
                  <div style={{ fontFamily: "monospace", fontSize: 12, wordBreak: "break-all" }}>{selectedNode.id}</div>
                </div>

                <div style={{ marginBottom: 10 }}>
                  <span style={{ fontSize: 11, color: "var(--text-dim)" }}>Fact Content</span>
                  <div style={{ fontSize: 13, marginTop: 2, lineHeight: 1.4 }}>{selectedNode.text}</div>
                </div>

                <div style={{ marginBottom: 10 }}>
                  <span style={{ fontSize: 11, color: "var(--text-dim)" }}>Status</span>
                  <div>
                    <span className={`badge ${selectedNode.status === "ACTIVE" ? "badge-active" : ""}`}>
                      {selectedNode.status}
                    </span>
                  </div>
                </div>

                {selectedNode.created_at && (
                  <div style={{ marginBottom: 10 }}>
                    <span style={{ fontSize: 11, color: "var(--text-dim)" }}>Recorded</span>
                    <div style={{ fontSize: 12 }}>{selectedNode.created_at}</div>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ fontSize: 12, color: "var(--text-faint)", marginTop: 20 }}>
                Click on any fact node to inspect its Neo4j graph metadata and relationship lineage.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
