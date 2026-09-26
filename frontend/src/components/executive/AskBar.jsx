import { useState } from "react";

const SUGGESTIONS = [
  "What database are we currently using?",
  "What database did we use before?",
  "What does Project Alpha depend on?",
  "What authentication method are we using?",
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
          placeholder='Ask e.g. "What database are we currently using?"'
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
