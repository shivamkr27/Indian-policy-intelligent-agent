import { useEffect, useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import MessageBubble from "./components/MessageBubble";
import ChatInput from "./components/ChatInput";
import UploadModal from "./components/UploadModal";
import MemoriesModal from "./components/MemoriesModal";
import {
  deleteConversation,
  fetchConversations,
  fetchDocuments,
  fetchMemories,
  ingestAllDocuments,
  streamChat,
  submitFeedback,
  uploadDocuments,
} from "./api";
import type { ChatMessage, ConversationSummary, MemoryItem } from "./types";

function newThreadId(): string {
  return crypto.randomUUID();
}

export default function App() {
  const [threadId, setThreadId] = useState(newThreadId);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [documents, setDocuments] = useState<string[]>([]);
  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [answerLanguage, setAnswerLanguage] = useState<"english" | "hindi">("english");
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [showMemories, setShowMemories] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);

  const refreshConversations = () => fetchConversations().then(setConversations).catch(() => {});
  const refreshDocuments = () => fetchDocuments().then(setDocuments).catch(() => {});

  useEffect(() => {
    refreshConversations();
    refreshDocuments();
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const handleNewChat = () => {
    setThreadId(newThreadId());
    setMessages([]);
  };

  const handleSelectConversation = (id: string) => {
    setThreadId(id);
    setMessages([]);
  };

  const handleDeleteConversation = async (id: string) => {
    await deleteConversation(id);
    if (id === threadId) handleNewChat();
    refreshConversations();
  };

  const handleSend = async (text: string) => {
    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", content: text };
    const assistantId = crypto.randomUUID();
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      isStreaming: true,
      toolSteps: [],
      reasoningSteps: [],
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);

    const update = (patch: Partial<ChatMessage>) => {
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, ...patch } : m))
      );
    };

    try {
      await streamChat(
        { threadId, message: text, answerLanguage, webSearchEnabled },
        (event) => {
          switch (event.type) {
            case "token":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, content: m.content + event.content } : m
                )
              );
              break;
            case "tool_start":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        toolSteps: [
                          ...(m.toolSteps ?? []),
                          { tool: event.tool, query: event.query, status: "running" },
                        ],
                      }
                    : m
                )
              );
              break;
            case "tool_end":
              setMessages((prev) =>
                prev.map((m) => {
                  if (m.id !== assistantId) return m;
                  const steps = [...(m.toolSteps ?? [])];
                  for (let i = steps.length - 1; i >= 0; i--) {
                    if (steps[i].tool === event.tool && steps[i].status === "running") {
                      steps[i] = { ...steps[i], status: "done", preview: event.preview };
                      break;
                    }
                  }
                  return { ...m, toolSteps: steps };
                })
              );
              break;
            case "reasoning_step":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        reasoningSteps: [
                          ...(m.reasoningSteps ?? []).filter((s) => s.step !== event.step),
                          { step: event.step, total: event.total, preview: event.preview },
                        ],
                      }
                    : m
                )
              );
              break;
            case "clarification":
              update({ content: event.question, isStreaming: false, isClarification: true });
              break;
            case "final":
              update({
                content: event.content,
                sources: event.sources,
                judgeBadge: event.judge_badge,
                judgeReason: event.judge_reason,
                isStreaming: false,
              });
              break;
            case "error":
              update({ content: `⚠️ ${event.message}`, isStreaming: false, isError: true });
              break;
          }
        }
      );
    } finally {
      setIsStreaming(false);
      refreshConversations();
      fetchMemories().then(setMemories).catch(() => {});
    }
  };

  const handleFeedback = (assistantContent: string, rating: "good" | "review" | "wrong") => {
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    submitFeedback(lastUser?.content ?? assistantContent, rating).catch(() => {});
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <Sidebar
        conversations={conversations}
        activeThreadId={threadId}
        documents={documents}
        onNewChat={handleNewChat}
        onSelectConversation={handleSelectConversation}
        onDeleteConversation={handleDeleteConversation}
        onUploadClick={() => setShowUpload(true)}
        onIngestAll={() => ingestAllDocuments().then(refreshDocuments)}
        onOpenMemories={() => {
          fetchMemories().then(setMemories).catch(() => {});
          setShowMemories(true);
        }}
      />

      <main className="flex-1 flex flex-col h-full">
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 pt-8">
          <div className="max-w-3xl mx-auto">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-[60vh] text-center">
                <h1 className="text-3xl font-medium bg-gradient-to-r from-accent to-accent-strong bg-clip-text text-transparent">
                  InsightEngine AI
                </h1>
                <p className="text-text-muted mt-2 text-sm">
                  Ask anything from your documents, or toggle web search for live information.
                </p>
              </div>
            )}
            {messages.map((m) => (
              <MessageBubble
                key={m.id}
                message={m}
                onFeedback={
                  m.role === "assistant" && !m.isStreaming && !m.isClarification && !m.isError
                    ? (rating) => handleFeedback(m.content, rating)
                    : undefined
                }
              />
            ))}
          </div>
        </div>

        <ChatInput
          disabled={isStreaming}
          answerLanguage={answerLanguage}
          webSearchEnabled={webSearchEnabled}
          onToggleLanguage={() => setAnswerLanguage((l) => (l === "english" ? "hindi" : "english"))}
          onToggleWebSearch={() => setWebSearchEnabled((v) => !v)}
          onUploadClick={() => setShowUpload(true)}
          onSend={handleSend}
        />
      </main>

      {showUpload && (
        <UploadModal
          onClose={() => {
            setShowUpload(false);
            refreshDocuments();
          }}
          onUpload={uploadDocuments}
        />
      )}

      {showMemories && <MemoriesModal memories={memories} onClose={() => setShowMemories(false)} />}
    </div>
  );
}
