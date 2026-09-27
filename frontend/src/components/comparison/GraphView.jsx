import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../../api";

/**
 * Neo4j Fact Knowledge Graph — animated, entity-clustered force layout
 * (docs/IMPLEMENTATION_V2.md §7). Fact nodes cluster around their :Entity's
 * position rather than floating as one undifferentiated cloud; new nodes fade
 * in rather than snapping into place, so "forming in the background" (the
 * decoupled graph write, §4.2) actually reads as live to a judge watching.
 * Stale facts stay in the graph (never deleted, only marked, per V1 §6.1) —
 * shown desaturated, with a dashed SUPERSEDED_BY edge to whatever replaced them.
 */

const WIDTH = 720;
const HEIGHT = 420;
const REPULSION = 2200;
const SPRING_K = 0.02;
const DAMPING = 0.82;
const FADE_MS = 700;
const NODE_R = 10;
const ENTITY_R = 14;

function layoutEntityCentroids(entityNames) {
  const cx = WIDTH / 2;
  const cy = HEIGHT / 2;
  const radius = Math.max(60, Math.min(WIDTH, HEIGHT) / 2 - 90);
  const n = entityNames.length || 1;
  const centroids = {};
  entityNames.forEach((name, i) => {
    const angle = (i / n) * 2 * Math.PI - Math.PI / 2;
    centroids[name] = { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
  });
  return centroids;
}

function entityIdToName(id) {
  return id && id.startsWith("entity::") ? id.slice("entity::".length) : null;
}

export default function GraphView({ sessionId, refreshSignal }) {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [, bumpFrame] = useState(0);

  const posRef = useRef(new Map()); // fact node id -> {x, y, vx, vy, bornAt}
  const rafRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    api
      .getGraphSnapshot(sessionId || "default_session")
      .then((data) => {
        if (!cancelled && data) {
          setGraphData({ nodes: data.nodes || [], edges: data.edges || [] });
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

  const factNodes = useMemo(() => graphData.nodes.filter((n) => n.label !== "Entity"), [graphData.nodes]);
  const entityNodes = useMemo(() => graphData.nodes.filter((n) => n.label === "Entity"), [graphData.nodes]);
  const entityNames = useMemo(() => entityNodes.map((n) => n.name).filter(Boolean).sort(), [entityNodes]);
  const entityKey = entityNames.join("|");
  const centroids = useMemo(() => layoutEntityCentroids(entityNames), [entityKey]);

  const activeNodes = factNodes.filter((n) => n.status !== "stale");
  const staleNodes = factNodes.filter((n) => n.status === "stale");

  // Seed positions for newly-arrived fact nodes near their entity's cluster
  // (small random offset, not the exact centroid, so overlapping new nodes
  // separate visibly via repulsion rather than starting stacked); keep
  // existing nodes' positions untouched across refetches so the layout
  // doesn't jump. `bornAt` drives the fade-in.
  useEffect(() => {
    const pos = posRef.current;
    const now = performance.now();
    const known = new Set();
    for (const node of factNodes) {
      known.add(node.id);
      if (!pos.has(node.id)) {
        const c = centroids[node.entity] || { x: WIDTH / 2, y: HEIGHT / 2 };
        pos.set(node.id, {
          x: c.x + (Math.random() - 0.5) * 30,
          y: c.y + (Math.random() - 0.5) * 30,
          vx: 0,
          vy: 0,
          bornAt: now,
        });
      }
    }
    for (const id of Array.from(pos.keys())) {
      if (!known.has(id)) pos.delete(id); // session reset / node no longer present
    }
  }, [factNodes, centroids]);

  // Continuous force simulation: pairwise repulsion between fact nodes +
  // a spring pulling each toward its own entity's centroid (the clustering
  // force) — this is what separates clusters instead of one blob (§7.1).
  useEffect(() => {
    function tick() {
      const pos = posRef.current;
      const ids = factNodes.map((n) => n.id);

      for (let i = 0; i < ids.length; i++) {
        const a = pos.get(ids[i]);
        if (!a) continue;
        for (let j = i + 1; j < ids.length; j++) {
          const b = pos.get(ids[j]);
          if (!b) continue;
          let dx = a.x - b.x;
          let dy = a.y - b.y;
          const distSq = Math.max(dx * dx + dy * dy, 4);
          const dist = Math.sqrt(distSq);
          const force = REPULSION / distSq;
          dx /= dist;
          dy /= dist;
          a.vx += dx * force;
          a.vy += dy * force;
          b.vx -= dx * force;
          b.vy -= dy * force;
        }
      }

      for (const node of factNodes) {
        const p = pos.get(node.id);
        if (!p) continue;
        const c = centroids[node.entity] || { x: WIDTH / 2, y: HEIGHT / 2 };
        p.vx += (c.x - p.x) * SPRING_K;
        p.vy += (c.y - p.y) * SPRING_K;
      }

      for (const node of factNodes) {
        const p = pos.get(node.id);
        if (!p) continue;
        p.vx *= DAMPING;
        p.vy *= DAMPING;
        p.x = Math.max(NODE_R + 4, Math.min(WIDTH - NODE_R - 4, p.x + p.vx));
        p.y = Math.max(NODE_R + 4, Math.min(HEIGHT - NODE_R - 4, p.y + p.vy));
      }

      bumpFrame((n) => (n + 1) % 1000000);
      rafRef.current = requestAnimationFrame(tick);
    }

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [factNodes, centroids]);

  function getPos(id) {
    const entityName = entityIdToName(id);
    if (entityName) return centroids[entityName] || { x: WIDTH / 2, y: HEIGHT / 2 };
    return posRef.current.get(id) || { x: WIDTH / 2, y: HEIGHT / 2 };
  }

  const now = performance.now();
  const opacityFor = (p) => Math.max(0, Math.min(1, (now - (p.bornAt ?? 0)) / FADE_MS));

  return (
    <div className="card graph-view-card">
      <div className="section-heading" style={{ marginBottom: 12 }}>
        <div>
          <h3 className="card-title" style={{ margin: 0 }}>Neo4j Fact Knowledge Graph (Agent 2 Memory)</h3>
          <p className="card-note" style={{ margin: "4px 0 0 0" }}>
            Facts cluster by entity; new writes from the decoupled background step (§4.2) fade in live.
            Stale facts stay visible (never deleted) with a dashed <code>SUPERSEDED_BY</code> edge to their replacement.
          </p>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span className="badge badge-active">{activeNodes.length} Active</span>
          <span className="badge" style={{ background: "rgba(255,107,107,0.15)", color: "var(--red)" }}>
            {staleNodes.length} Stale / Superseded
          </span>
        </div>
      </div>

      {loading && factNodes.length === 0 ? (
        <div className="empty-note">Loading Neo4j graph data...</div>
      ) : factNodes.length === 0 ? (
        <div className="empty-note">
          No facts stored yet. Run a prompt that mentions a fact (e.g. "We are migrating to PostgreSQL next week") to watch Agent 2 extract facts and form graph relationships.
        </div>
      ) : (
        <div className="graph-container-flex" style={{ display: "flex", gap: 16, minHeight: 340 }}>
          <div
            className="graph-canvas-box"
            style={{ flex: 1, background: "var(--panel-2)", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", overflow: "hidden" }}
          >
            <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} width="100%" height="100%" style={{ display: "block", minHeight: 340 }}>
              {/* Edges first, so nodes render on top */}
              {graphData.edges.map((edge, idx) => {
                const a = getPos(edge.source);
                const b = getPos(edge.target);
                const isSupersede = edge.type === "SUPERSEDED_BY";
                return (
                  <line
                    key={idx}
                    x1={a.x}
                    y1={a.y}
                    x2={b.x}
                    y2={b.y}
                    stroke={isSupersede ? "var(--amber)" : "var(--border)"}
                    strokeWidth={isSupersede ? 1.5 : 1}
                    strokeDasharray={isSupersede ? "5 4" : undefined}
                    opacity={isSupersede ? 0.8 : 0.5}
                  />
                );
              })}

              {/* Entity cluster centroids + labels */}
              {entityNodes.map((entity) => {
                const c = centroids[entity.name] || { x: WIDTH / 2, y: HEIGHT / 2 };
                return (
                  <g key={entity.id}>
                    <circle cx={c.x} cy={c.y} r={ENTITY_R} fill="var(--panel)" stroke="var(--text-dim)" strokeWidth={1.5} />
                    <text
                      x={c.x}
                      y={c.y - ENTITY_R - 6}
                      textAnchor="middle"
                      fontSize={11}
                      fontWeight={700}
                      fill="var(--text)"
                      style={{ textTransform: "uppercase", letterSpacing: 0.4 }}
                    >
                      {entity.name}
                    </text>
                  </g>
                );
              })}

              {/* Fact nodes */}
              {factNodes.map((node) => {
                const p = posRef.current.get(node.id);
                if (!p) return null;
                const isStale = node.status === "stale";
                const isSelected = selectedNode?.id === node.id;
                return (
                  <g
                    key={node.id}
                    transform={`translate(${p.x}, ${p.y})`}
                    opacity={opacityFor(p)}
                    onClick={() => setSelectedNode(node)}
                    style={{ cursor: "pointer" }}
                  >
                    <circle
                      r={isSelected ? NODE_R + 2 : NODE_R}
                      fill={isStale ? "var(--panel)" : "var(--green-dim)"}
                      stroke={isSelected ? "var(--green)" : isStale ? "var(--text-faint)" : "var(--green)"}
                      strokeWidth={isSelected ? 2.5 : 1.5}
                      strokeDasharray={isStale ? "3 2" : undefined}
                    />
                  </g>
                );
              })}
            </svg>
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
                  <span style={{ fontSize: 11, color: "var(--text-dim)" }}>Entity</span>
                  <div style={{ fontSize: 13, marginTop: 2 }}>{selectedNode.entity || "—"}</div>
                </div>

                <div style={{ marginBottom: 10 }}>
                  <span style={{ fontSize: 11, color: "var(--text-dim)" }}>Fact Content</span>
                  <div style={{ fontSize: 13, marginTop: 2, lineHeight: 1.4 }}>{selectedNode.text}</div>
                </div>

                <div style={{ marginBottom: 10 }}>
                  <span style={{ fontSize: 11, color: "var(--text-dim)" }}>Status</span>
                  <div>
                    <span className={`badge ${selectedNode.status !== "stale" ? "badge-active" : ""}`}>
                      {(selectedNode.status || "active").toUpperCase()}
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
                Click any fact node to inspect it. Dashed nodes/edges are stale facts and their <code>SUPERSEDED_BY</code> lineage — kept in the graph, excluded from Agent 2's context.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
