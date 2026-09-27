export default function Header({ view, onChangeView }) {
  return (
    <header className="app-header">
      <div className="brand">
        <div className="logo">
          MEM<em>TRACE</em>
        </div>
      </div>

      <div className="header-controls">
        <div className="view-toggle">
          <button className={view === "comparison" ? "active" : ""} onClick={() => onChangeView("comparison")}>
            Agent 1 vs Agent 2 (Live Benchmark)
          </button>
          <button className={view === "executive" ? "active" : ""} onClick={() => onChangeView("executive")}>
            Executive View
          </button>
        </div>
      </div>
    </header>
  );
}

