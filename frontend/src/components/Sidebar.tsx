import type { ConversationSummary } from "../types";

export default function Sidebar({
  conversations,
  activeThreadId,
  documents,
  onNewChat,
  onSelectConversation,
  onDeleteConversation,
  onUploadClick,
  onIngestAll,
  onOpenMemories,
}: {
  conversations: ConversationSummary[];
  activeThreadId: string;
  documents: string[];
  onNewChat: () => void;
  onSelectConversation: (threadId: string) => void;
  onDeleteConversation: (threadId: string) => void;
  onUploadClick: () => void;
  onIngestAll: () => void;
  onOpenMemories: () => void;
}) {
  return (
    <aside className="w-72 shrink-0 h-full bg-bg-elevated border-r border-border flex flex-col">
      <div className="p-3">
        <button
          onClick={onNewChat}
          className="w-full flex items-center gap-2 rounded-full border border-border-strong px-4 py-2.5 text-sm text-text hover:bg-surface transition-colors"
        >
          <span className="text-accent text-lg leading-none">+</span> New chat
        </button>
      </div>

      <div className="px-3 pb-2">
        <div className="flex items-center justify-between text-xs font-medium text-text-muted uppercase tracking-wide px-1 mb-1.5">
          <span>Documents ({documents.length})</span>
          <div className="flex gap-2">
            <button onClick={onUploadClick} className="hover:text-accent transition-colors" title="Upload">
              📎
            </button>
            <button onClick={onIngestAll} className="hover:text-accent transition-colors" title="Re-ingest all">
              🔄
            </button>
          </div>
        </div>
        <div className="max-h-28 overflow-y-auto flex flex-col gap-0.5">
          {documents.length === 0 && (
            <div className="text-xs text-text-muted px-1 py-1">No documents yet.</div>
          )}
          {documents.map((doc) => (
            <div
              key={doc}
              className="text-xs text-text-secondary px-2 py-1 rounded-lg truncate"
              title={doc}
            >
              {doc.endsWith(".pdf") ? "📕" : doc.endsWith(".docx") ? "📘" : "📄"} {doc}
            </div>
          ))}
        </div>
      </div>

      <div className="h-px bg-border mx-3 my-1" />

      <div className="flex-1 overflow-y-auto px-3 py-2">
        <div className="text-xs font-medium text-text-muted uppercase tracking-wide px-1 mb-1.5">
          Recent
        </div>
        <div className="flex flex-col gap-0.5">
          {conversations.map((c) => (
            <div
              key={c.thread_id}
              onClick={() => onSelectConversation(c.thread_id)}
              className={`group flex items-center justify-between gap-2 px-3 py-2 rounded-xl text-sm cursor-pointer transition-colors ${
                c.thread_id === activeThreadId
                  ? "bg-surface text-text"
                  : "text-text-secondary hover:bg-surface/60"
              }`}
            >
              <span className="truncate flex-1">{c.title}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteConversation(c.thread_id);
                }}
                className="opacity-0 group-hover:opacity-100 text-text-muted hover:text-danger transition-opacity shrink-0"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      </div>

      <div className="p-3 border-t border-border flex flex-col gap-1">
        <button
          onClick={onOpenMemories}
          className="text-left text-sm text-text-secondary hover:text-text px-3 py-2 rounded-xl hover:bg-surface transition-colors"
        >
          🧠 What I remember
        </button>
      </div>
    </aside>
  );
}
