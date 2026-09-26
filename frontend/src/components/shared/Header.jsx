export default function Header({ view, onChangeView, onSeed, seeding }) {
  return (
    <header className="app-header">
      <div className="brand">
        <div className="logo">
          MEM<em>TRACE</em>
        </div>
        <div className="tagline">AI memory that pays for itself</div>
      </div>

      <div className="header-controls">
        <div className="view-toggle">
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
