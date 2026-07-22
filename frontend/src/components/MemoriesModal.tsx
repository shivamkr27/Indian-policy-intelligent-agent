import Modal from "./Modal";
import type { MemoryItem } from "../types";

const TYPE_LABEL: Record<string, string> = {
  topic_interest: "Interest",
  preference: "Preference",
  knowledge_gap: "Knowledge gap",
  doc_affinity: "Document",
};

export default function MemoriesModal({
  memories,
  onClose,
}: {
  memories: MemoryItem[];
  onClose: () => void;
}) {
  return (
    <Modal title="What I remember about you" onClose={onClose}>
      {memories.length === 0 ? (
        <p className="text-sm text-text-muted">
          No memories yet — they build up as you chat and give feedback.
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          {memories.map((m, i) => (
            <div key={i} className="border border-border rounded-xl px-3 py-2.5">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium text-accent">
                  {TYPE_LABEL[m.memory_type] ?? m.memory_type}
                </span>
                <span className="text-xs text-text-muted">{"★".repeat(m.importance)}</span>
              </div>
              <p className="text-sm text-text-secondary">{m.content}</p>
            </div>
          ))}
        </div>
      )}
    </Modal>
  );
}
