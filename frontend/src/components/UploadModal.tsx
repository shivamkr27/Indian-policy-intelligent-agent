import { useRef, useState } from "react";
import Modal from "./Modal";
import type { DocumentUploadResult } from "../types";

export default function UploadModal({
  onClose,
  onUpload,
}: {
  onClose: () => void;
  onUpload: (files: File[]) => Promise<DocumentUploadResult[]>;
}) {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState<DocumentUploadResult[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const doUpload = async (files: FileList | File[]) => {
    const list = Array.from(files).slice(0, 5);
    if (!list.length) return;
    setUploading(true);
    try {
      const res = await onUpload(list);
      setResults(res);
    } finally {
      setUploading(false);
    }
  };

  return (
    <Modal title="Upload documents" onClose={onClose}>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          doUpload(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
          dragActive ? "border-accent bg-accent-dim" : "border-border hover:border-border-strong"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.txt"
          className="hidden"
          onChange={(e) => e.target.files && doUpload(e.target.files)}
        />
        <p className="text-sm text-text-secondary">
          {uploading ? "Uploading..." : "Drop files here, or click to browse"}
        </p>
        <p className="text-xs text-text-muted mt-1">PDF, DOCX, or TXT — up to 5 files</p>
      </div>

      {results.length > 0 && (
        <div className="mt-4 flex flex-col gap-1.5">
          {results.map((r, i) => (
            <div key={i} className="text-xs flex items-center gap-2">
              <span>{r.status === "ingested" ? "✅" : "❌"}</span>
              <span className="text-text-secondary">{r.file}</span>
              {r.status === "error" && <span className="text-danger">— {r.detail}</span>}
            </div>
          ))}
        </div>
      )}
    </Modal>
  );
}
