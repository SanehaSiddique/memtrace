import { formatUsd } from "../../format";

const COLORS = {
  outdated_information: "#f5b942",
  unverified_information: "#a78bfa",
  redundant_memory: "#5b9dff",
  irrelevant_context: "#ff6b6b",
};

export default function MoneyLeaks({ leaks }) {
  if (!leaks || leaks.length === 0) {
    return <div className="empty-note">No wasted spend detected yet in recorded activity — ask a few questions first.</div>;
  }
  return (
    <div className="grid grid-4">
      {leaks.map((leak) => (
        <div key={leak.category} className="leak-card">
          <div className="label">{leak.label}</div>
          <div className="amount">{formatUsd(leak.amount)}</div>
          <div className="leak-bar">
            <div style={{ width: `${Math.min(100, leak.percent)}%`, background: COLORS[leak.category] || "#888" }} />
          </div>
          <div className="leak-percent">{leak.percent.toFixed(1)}% of avoided spend</div>
        </div>
      ))}
    </div>
  );
}
