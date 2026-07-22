import { useRef, useState, type KeyboardEvent } from "react";

export default function ChatInput({
  disabled,
  answerLanguage,
  webSearchEnabled,
  onToggleLanguage,
  onToggleWebSearch,
  onUploadClick,
  onSend,
}: {
  disabled: boolean;
  answerLanguage: "english" | "hindi";
  webSearchEnabled: boolean;
  onToggleLanguage: () => void;
  onToggleWebSearch: () => void;
  onUploadClick: () => void;
  onSend: (message: string) => void;
}) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const autoGrow = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  };

  return (
    <div className="px-4 pb-4 pt-2">
      <div className="max-w-3xl mx-auto">
        <div className="flex items-end gap-2 bg-surface border border-border rounded-3xl px-4 py-2.5 focus-within:border-border-strong transition-colors">
          <button
            onClick={onUploadClick}
            title="Upload document"
            className="shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-text-secondary hover:bg-surface-hover hover:text-text transition-colors"
          >
            📎
          </button>

          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
              autoGrow();
            }}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything about your documents..."
            rows={1}
            className="flex-1 resize-none bg-transparent outline-none text-text placeholder:text-text-muted py-1.5 max-h-40"
          />

          <button
            onClick={handleSend}
            disabled={disabled || !value.trim()}
            title="Send"
            className="shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-accent text-bg disabled:bg-surface-hover disabled:text-text-muted transition-colors"
          >
            ➤
          </button>
        </div>

        <div className="flex items-center gap-2 mt-2 px-1">
          <button
            onClick={onToggleLanguage}
            className={`text-xs rounded-full px-3 py-1 border transition-colors ${
              answerLanguage === "hindi"
                ? "border-accent/40 bg-accent-dim text-accent"
                : "border-border text-text-secondary hover:border-border-strong"
            }`}
          >
            🇮🇳 Hindi {answerLanguage === "hindi" ? "ON" : "OFF"}
          </button>
          <button
            onClick={onToggleWebSearch}
            className={`text-xs rounded-full px-3 py-1 border transition-colors ${
              webSearchEnabled
                ? "border-accent/40 bg-accent-dim text-accent"
                : "border-border text-text-secondary hover:border-border-strong"
            }`}
          >
            🌐 Web Search {webSearchEnabled ? "ON" : "OFF"}
          </button>
        </div>
      </div>
    </div>
  );
}
