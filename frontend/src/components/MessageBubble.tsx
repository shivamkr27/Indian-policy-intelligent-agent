import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "../types";

function JudgeBadge({ badge, reason }: { badge: string; reason?: string }) {
  const isGreen = badge.includes("🟢");
  const isRed = badge.includes("🔴");
  const color = isGreen ? "text-success" : isRed ? "text-danger" : "text-warning";
  const bg = isGreen
    ? "bg-success/10 border-success/25"
    : isRed
    ? "bg-danger/10 border-danger/25"
    : "bg-warning/10 border-warning/25";

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className={`text-xs font-medium rounded-full border px-2.5 py-0.5 ${color} ${bg}`}>
        {badge}
      </span>
      {reason && <span className="text-xs text-text-muted">{reason}</span>}
    </div>
  );
}

function ToolStepsView({ message }: { message: ChatMessage }) {
  if (!message.toolSteps?.length && !message.reasoningSteps?.length) return null;
  return (
    <div className="mb-3 flex flex-col gap-1.5">
      {message.toolSteps?.map((step, i) => (
        <div
          key={i}
          className="text-xs text-text-secondary border border-border rounded-lg px-3 py-2 bg-surface/60"
        >
          <span className="text-accent">
            {step.tool === "web_search" ? "🌐" : "🔍"} {step.status === "running" ? "Searching" : "Searched"}:
          </span>{" "}
          {step.query}
          {step.preview && (
            <div className="mt-1 text-text-muted line-clamp-2">{step.preview}</div>
          )}
        </div>
      ))}
      {message.reasoningSteps?.map((step, i) => (
        <div
          key={`r${i}`}
          className="text-xs text-text-secondary border border-border rounded-lg px-3 py-2 bg-surface/60"
        >
          <span className="text-accent">
            🔗 Step {step.step}/{step.total}
          </span>
          {step.preview && (
            <div className="mt-1 text-text-muted line-clamp-2">{step.preview}</div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function MessageBubble({
  message,
  onFeedback,
}: {
  message: ChatMessage;
  onFeedback?: (rating: "good" | "review" | "wrong") => void;
}) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end mb-6">
        <div className="max-w-[75%] rounded-2xl rounded-br-sm bg-user-bubble px-4 py-2.5 text-text">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="mb-8 max-w-[85%]">
      <ToolStepsView message={message} />

      {message.content && (
        <div
          className={`prose prose-invert prose-sm max-w-none prose-p:leading-relaxed prose-pre:bg-surface prose-pre:border prose-pre:border-border ${
            message.isClarification ? "text-warning" : message.isError ? "text-danger" : "text-text"
          }`}
        >
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
        </div>
      )}

      {message.isStreaming && !message.content && (
        <div className="flex gap-1 py-1">
          <span className="w-1.5 h-1.5 rounded-full bg-accent animate-bounce [animation-delay:-0.3s]" />
          <span className="w-1.5 h-1.5 rounded-full bg-accent animate-bounce [animation-delay:-0.15s]" />
          <span className="w-1.5 h-1.5 rounded-full bg-accent animate-bounce" />
        </div>
      )}

      {!message.isStreaming && !message.isClarification && !message.isError && message.judgeBadge && (
        <div className="mt-3 pt-2.5 border-t border-border/60 flex flex-col gap-2.5">
          <JudgeBadge badge={message.judgeBadge} reason={message.judgeReason} />

          {message.sources && message.sources.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {message.sources.map((s, i) => (
                <span
                  key={i}
                  className="text-xs text-accent bg-accent-dim border border-accent/25 rounded px-2 py-0.5"
                >
                  📄 {s}
                </span>
              ))}
            </div>
          )}

          {onFeedback && (
            <div className="flex gap-2">
              <button
                onClick={() => onFeedback("good")}
                className="text-xs text-text-secondary border border-border rounded-md px-2.5 py-1 hover:border-success/50 hover:text-success transition-colors"
              >
                ✅ Accurate
              </button>
              <button
                onClick={() => onFeedback("review")}
                className="text-xs text-text-secondary border border-border rounded-md px-2.5 py-1 hover:border-warning/50 hover:text-warning transition-colors"
              >
                ⚠️ Partly right
              </button>
              <button
                onClick={() => onFeedback("wrong")}
                className="text-xs text-text-secondary border border-border rounded-md px-2.5 py-1 hover:border-danger/50 hover:text-danger transition-colors"
              >
                🚫 Wrong
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
