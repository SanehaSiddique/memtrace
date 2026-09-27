import { useState } from "react";

// Generic prompts, not facts from a demo story. They have to work against
// whatever the agent has actually been told, so none of them assume a
// particular datastore, vendor, or project.
const SUGGESTIONS = [
  "What do we currently use?",
  "What did we use before?",
  "What does the system depend on?",
  "What changed most recently?",
];

export default function AskBar({ onAsk, onTeach, busy }) {
  const [text, setText] = useState("");

  function submit(action) {
    const value = text.trim();
    if (!value) return;
    action(value);
  }

  return (
    <div>
      <div className="ask-bar">
        <input
          type="text"
          placeholder='Ask anything — "what are we using right now?"'
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit(onAsk)}
        />
        <button className="btn btn-primary" disabled={busy} onClick={() => submit(onAsk)}>
          Ask
        </button>
        <button className="btn" disabled={busy} onClick={() => submit(onTeach)}>
          Tell the agent
        </button>
      </div>
      <div className="ask-suggestions">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => {
              setText(s);
              onAsk(s);
            }}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
