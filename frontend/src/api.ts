import type {
  ChatStreamEvent,
  ConversationSummary,
  DocumentUploadResult,
  MemoryItem,
} from "./types";

const BASE = "/api";

export async function fetchConversations(): Promise<ConversationSummary[]> {
  const res = await fetch(`${BASE}/conversations`);
  const data = await res.json();
  return data.conversations;
}

export async function fetchConversation(
  threadId: string
): Promise<{ role: "user" | "assistant"; content: string }[]> {
  const res = await fetch(`${BASE}/conversations/${encodeURIComponent(threadId)}`);
  const data = await res.json();
  return data.messages;
}

export async function deleteConversation(threadId: string): Promise<void> {
  await fetch(`${BASE}/conversations/${encodeURIComponent(threadId)}`, { method: "DELETE" });
}

export async function fetchDocuments(): Promise<string[]> {
  const res = await fetch(`${BASE}/documents`);
  const data = await res.json();
  return data.files;
}

export async function uploadDocuments(files: File[]): Promise<DocumentUploadResult[]> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${BASE}/documents/upload`, { method: "POST", body: form });
  const data = await res.json();
  return data.results;
}

export async function ingestAllDocuments(): Promise<void> {
  await fetch(`${BASE}/documents/ingest-all`, { method: "POST" });
}

export async function fetchMemories(): Promise<MemoryItem[]> {
  const res = await fetch(`${BASE}/memories`);
  const data = await res.json();
  return data.memories;
}

export async function submitFeedback(
  question: string,
  rating: "good" | "review" | "wrong"
): Promise<void> {
  await fetch(`${BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, rating }),
  });
}

/**
 * Streams a chat turn via SSE over a POST request. Native EventSource can't
 * send a POST body, so this reads the fetch() response body manually and
 * splits it on the SSE "\n\n" record separator.
 */
export async function streamChat(
  params: {
    threadId: string;
    message: string;
    answerLanguage: "english" | "hindi";
    webSearchEnabled: boolean;
  },
  onEvent: (event: ChatStreamEvent) => void
): Promise<void> {
  const res = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      thread_id: params.threadId,
      message: params.message,
      answer_language: params.answerLanguage,
      web_search_enabled: params.webSearchEnabled,
    }),
  });

  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => "");
    onEvent({ type: "error", message: detail || `Request failed (${res.status})` });
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const records = buffer.split("\n\n");
    buffer = records.pop() ?? "";

    for (const record of records) {
      const line = record.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      try {
        const event = JSON.parse(line.slice("data: ".length)) as ChatStreamEvent;
        onEvent(event);
      } catch {
        // ignore malformed chunk
      }
    }
  }
}
