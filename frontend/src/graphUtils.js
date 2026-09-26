// Shared helpers for turning real /memory/graph-full data into rendered
// chains. Kept out of components so MemoryGraph.jsx and LiveMemoryDemo.jsx
// don't duplicate (or drift on) the same lineage-walking logic.

export const RELATION_LABELS = {
  REPLACED_BY: "replaced by",
  SUPERSEDES: "supersedes",
  DEPENDS_ON: "depends on",
  CAUSED_BY: "caused by",
  RELATED_TO: "related to",
};

// Walk each memory's real `supersedes_memory_id` chain (set by the backend
// on every UPDATE) back to its origin. A "head" is any memory nothing has
// replaced yet — it's the current end of its lineage, whatever its status.
export function buildChains(nodes) {
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const heads = nodes.filter((n) => !n.superseded_by_memory_id);

  return heads.map((head) => {
    const lineage = [head];
    let current = head;
    while (current.supersedes_memory_id && byId[current.supersedes_memory_id]) {
      current = byId[current.supersedes_memory_id];
      lineage.push(current);
    }
    return lineage; // [newest, ..., oldest]
  });
}

export function edgeRelationLabel(edges, oldId, newId) {
  const edge = edges.find((e) => e.source_id === oldId && e.target_id === newId);
  return edge ? RELATION_LABELS[edge.relation_type] || edge.relation_type.toLowerCase() : "replaced by";
}
