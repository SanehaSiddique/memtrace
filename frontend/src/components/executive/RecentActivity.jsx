import { formatUsd } from "../../format";

export default function RecentActivity({ runs }) {
  if (!runs || runs.length === 0) {
    return <div className="empty-note">No activity yet — ask a question above to see it recorded here.</div>;
  }
  return (
    <div className="feed-list">
      {runs.map((r) => (
        <div className="feed-row" key={r.id}>
          <span className="q">{r.query}</span>
          <span className="saved">saved {formatUsd(r.savings, { decimals: 4 })}</span>
        </div>
      ))}
    </div>
  );
}
