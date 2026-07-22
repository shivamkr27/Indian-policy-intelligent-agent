export type Role = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  sources?: string[];
  judgeBadge?: string;
  judgeReason?: string;
  toolSteps?: ToolStep[];
  reasoningSteps?: ReasoningStepInfo[];
  isStreaming?: boolean;
  isClarification?: boolean;
  isError?: boolean;
}

export interface ToolStep {
  tool: string;
  query?: string;
  preview?: string;
  status: "running" | "done";
}

export interface ReasoningStepInfo {
  step: number;
  total: number;
  preview: string;
}

export interface ConversationSummary {
  thread_id: string;
  title: string;
  created_at: string;
  last_active: string;
  message_count: number;
}

export interface DocumentUploadResult {
  file: string;
  status: "ingested" | "error";
  detail?: string;
  parent_chunks?: number;
  child_chunks?: number;
}

export interface MemoryItem {
  content: string;
  memory_type: string;
  importance: number;
  timestamp: string;
}

// SSE event payloads from POST /api/chat/stream
export type ChatStreamEvent =
  | { type: "token"; content: string }
  | { type: "tool_start"; tool: string; query: string }
  | { type: "tool_end"; tool: string; preview: string }
  | { type: "reasoning_plan"; steps: string[] }
  | { type: "reasoning_step"; step: number; total: number; preview: string }
  | { type: "clarification"; question: string }
  | {
      type: "final";
      content: string;
      sources: string[];
      judge_badge: string;
      judge_reason: string;
    }
  | { type: "error"; message: string };
