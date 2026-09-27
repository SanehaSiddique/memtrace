export default function Header({ view, onChangeView, onSeed, seeding }) {
  return (
    <header className="app-header">
      <div className="brand">
        <div className="logo">
          MEM<em>TRACE</em>
        </div>
        <div className="tagline">Dual-Agent Benchmark & Memory Optimization</div>
      </div>

      <div className="header-controls">
        <div className="view-toggle">
          <button className={view === "comparison" ? "active" : ""} onClick={() => onChangeView("comparison")}>
            Agent 1 vs Agent 2 (Live Benchmark)
          </button>
          <button className={view === "executive" ? "active" : ""} onClick={() => onChangeView("executive")}>
            Executive View
          </button>
          <button className={view === "engineering" ? "active" : ""} onClick={() => onChangeView("engineering")}>
            Engineering View
          </button>
        </div>
        <button className="btn" onClick={onSeed} disabled={seeding}>
          {seeding ? "Seeding…" : "Seed demo data"}
        </button>
      </div>
    </header>
  );
}

