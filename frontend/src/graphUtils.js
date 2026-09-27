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

// The taxonomy JEV picks from when it retires a fact (mirrors
// STALE_CLASSES in backend/app/agent2/graph_cleaning.py). Duplicated here on
// purpose: the UI must render a label even for nodes classified by an older
// backend, and the backend must not be imported into the browser bundle.
export const STALE_CLASS_META = {
  outdated: { label: "Outdated", color: "#f5b942", desc: "Was true, but the situation has since changed." },
  corrected: { label: "Corrected", color: "#a855f7", desc: "A newer fact fixed an error in the older one." },
  preference_change: { label: "Preference change", color: "#5b8cff", desc: "The subject changed their mind or choice." },
  duplicate: { label: "Duplicate", color: "#64646f", desc: "Restated by a newer fact in different words." },
  scope_change: { label: "Scope change", color: "#3ddc84", desc: "The newer fact narrowed or widened the claim." },
  different_subject: { label: "Different subject", color: "#ff6b6b", desc: "Looked like a conflict, but wasn't one." },
  unclassified: { label: "Unclassified", color: "#64646f", desc: "JEV could not confidently say why." },
};

export const UNCLASSIFIED = "unclassified";

export function classMeta(key) {
  return STALE_CLASS_META[key] || STALE_CLASS_META[UNCLASSIFIED];
}

/**
 * Group stale fact nodes by the class JEV assigned them, largest group first.
 * Active nodes are returned separately — they have no class, since "why did
 * this lose" only applies to nodes that lost.
 */
export function groupByStaleClass(nodes) {
  const stale = nodes.filter((n) => n.status === "stale");
  const active = nodes.filter((n) => n.status !== "stale");

  const groups = new Map();
  for (const node of stale) {
    const key = node.stale_class || UNCLASSIFIED;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(node);
  }

  return {
    active,
    groups: [...groups.entries()]
      .map(([key, items]) => ({ key, items, meta: classMeta(key) }))
      .sort((a, b) => b.items.length - a.items.length),
  };
}
