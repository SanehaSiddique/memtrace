// Credibility rule: every metric on the Executive View must say plainly
// whether it's an ACTUAL count, a CALCULATED figure (real activity x
// configured pricing), a PROJECTED extrapolation, or DEMO data.
const STYLES = {
  ACTUAL: { color: "var(--green)", background: "var(--green-dim)" },
  CALCULATED: { color: "var(--blue)", background: "var(--blue-dim)" },
  PROJECTED: { color: "var(--amber)", background: "var(--amber-dim)" },
  DEMO: { color: "var(--purple)", background: "var(--purple-dim)" },
};

export default function MetricBadge({ kind }) {
  const style = STYLES[kind] || STYLES.DEMO;
  return (
    <span
      style={{
        display: "inline-block",
        fontSize: 10,
        fontWeight: 800,
        letterSpacing: 0.5,
        textTransform: "uppercase",
        padding: "2px 7px",
        borderRadius: 999,
        marginLeft: 8,
        verticalAlign: "middle",
        ...style,
      }}
    >
      {kind}
    </span>
  );
}
