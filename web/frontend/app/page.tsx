"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Send, Image as ImageIcon, X, Loader2, ChevronDown,
  Zap, Brain, Eye, ScanEye, Plus, Sparkles, Check
} from "lucide-react";

// ── Types ──────────────────────────────────────────────────────
type Mode = "eff" | "swin" | "xai-eff" | "xai-swin";

type XAIModelData = {
  grade: number;
  raw_score: number;
  description: string;
  heatmap_b64: string;
  overlay_b64: string;
};

type SingleResult = {
  mode: Mode;
  filename: string;
  ai_answer: string;
  grade?: number;
  raw_score?: number;
  description?: string;
  original_b64?: string;
  efficientnet?: XAIModelData;
  swinv2?: XAIModelData;
};

type Message = {
  id: number;
  role: "user" | "bot";
  text?: string;
  previewUrls?: string[];
  results?: SingleResult[];
  isError?: boolean;
};

// ── Grade palette ──────────────────────────────────────────────
const GRADE_META: Record<number, { color: string; bg: string; label: string }> = {
  0: { color: "#16a34a", bg: "#dcfce7", label: "No DR" },
  1: { color: "#65a30d", bg: "#ecfccb", label: "Mild" },
  2: { color: "#d97706", bg: "#fef3c7", label: "Moderate" },
  3: { color: "#ea580c", bg: "#ffedd5", label: "Severe" },
  4: { color: "#dc2626", bg: "#fee2e2", label: "Proliferative" },
};

// ── GradeBadge ─────────────────────────────────────────────────
function GradeBadge({ grade }: { grade: number }) {
  const m = GRADE_META[grade] ?? GRADE_META[0];
  return (
    <span
      className="inline-flex items-center gap-1 text-[11px] font-semibold px-2 py-0.5 rounded-full tracking-wide"
      style={{ color: m.color, background: m.bg }}
    >
      G{grade} · {m.label}
    </span>
  );
}

// ── XAIModelCard ───────────────────────────────────────────────
function XAIModelCard({ title, data, accent }: { title: string; data: XAIModelData; accent: string }) {
  const [tab, setTab] = useState<"overlay" | "heatmap">("overlay");
  return (
    <div className="flex-1 min-w-0 rounded-2xl border overflow-hidden"
         style={{ borderColor: accent + "30", background: "#fafaf9" }}>
      <div className="px-3 py-2 flex items-center justify-between border-b"
           style={{ borderColor: accent + "20", background: accent + "08" }}>
        <span className="text-[11px] font-semibold tracking-wide" style={{ color: accent }}>{title}</span>
        <GradeBadge grade={data.grade} />
      </div>
      <div className="flex border-b border-[#e8e5e0]">
        {(["overlay", "heatmap"] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)}
            className="flex-1 text-[11px] py-1.5 font-medium transition-colors"
            style={{
              color: tab === t ? accent : "#9ca3af",
              borderBottom: tab === t ? `2px solid ${accent}` : "2px solid transparent",
              background: tab === t ? accent + "08" : "transparent",
            }}>
            {t === "overlay" ? "CAM Overlay" : "Heatmap"}
          </button>
        ))}
      </div>
      <img src={`data:image/jpeg;base64,${tab === "overlay" ? data.overlay_b64 : data.heatmap_b64}`}
           alt={`${title} ${tab}`} className="w-full object-cover" />
      <div className="px-3 py-2">
        <p className="text-xs text-[#4a4540] leading-snug">{data.description}</p>
        <p className="text-[10px] text-[#a09890] mt-1 font-mono">score: {data.raw_score}</p>
      </div>
    </div>
  );
}

// ── ResultCard ─────────────────────────────────────────────────
function ResultCard({ result, index }: { result: SingleResult; index: number }) {
  const [open, setOpen] = useState(true);
  const displayGrade = result.grade ?? result.efficientnet?.grade ?? result.swinv2?.grade;

  return (
    <div className="rounded-2xl border border-[#e8e5e0] overflow-hidden bg-white">
      <button onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-[#faf9f7] transition-colors">
        <span className="text-xs font-medium text-[#6b6560] truncate max-w-[60%]">
          {index + 1}. {result.filename}
        </span>
        <div className="flex items-center gap-2 shrink-0">
          {displayGrade !== undefined && <GradeBadge grade={displayGrade} />}
          <ChevronDown className={`w-3.5 h-3.5 text-[#a09890] transition-transform ${open ? "rotate-180" : ""}`} />
        </div>
      </button>

      {open && (
        <div className="px-4 pb-4 space-y-3 border-t border-[#f0ede8]">
          {(result.mode === "eff" || result.mode === "swin") && (
            <p className="text-sm text-[#1c1917] leading-relaxed pt-3">{result.ai_answer}</p>
          )}
          {result.mode.startsWith("xai") && (
            <>
              {result.original_b64 && (
                <div className="flex items-start gap-3 pt-3">
                  <img src={`data:image/jpeg;base64,${result.original_b64}`} alt="original"
                    className="h-14 w-14 rounded-xl object-cover border border-[#e8e5e0] shrink-0" />
                  <p className="text-sm text-[#1c1917] leading-relaxed">{result.ai_answer}</p>
                </div>
              )}
              <div className="flex gap-2">
                {result.efficientnet && (result.mode === "xai-eff") && (
                  <XAIModelCard title="EfficientNet-B7" data={result.efficientnet} accent="#c2640a" />
                )}
                {result.swinv2 && (result.mode === "xai-swin") && (
                  <XAIModelCard title="SwinV2-Base" data={result.swinv2} accent="#6d28d9" />
                )}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ── Mode config ────────────────────────────────────────────────
const MODES: { id: Mode; label: string; sub: string; desc: string; icon: React.ReactNode }[] = [
  { id: "eff",      label: "EfficientNet",  sub: "B7",      desc: "Trích xuất đặc trưng với mạng CNN chuẩn", icon: <Zap className="w-3.5 h-3.5" /> },
  { id: "swin",     label: "SwinV2",        sub: "Base",    desc: "Giải quyết các vấn đề phức tạp với Transformer", icon: <Brain className="w-3.5 h-3.5" /> },
  { id: "xai-eff",  label: "XAI",           sub: "Eff",     desc: "Phân tích chuẩn đoán và xuất bản đồ nhiệt CNN", icon: <Eye className="w-3.5 h-3.5" /> },
  { id: "xai-swin", label: "XAI",           sub: "Swin",    desc: "Giải thích quyết định chuyên sâu với Transformer", icon: <ScanEye className="w-3.5 h-3.5" /> },
];

// ── Dropdown Component (Giống menu của Gemini) ──────────────────
function ModelDropdown({ mode, onChange }: { mode: Mode; onChange: (m: Mode) => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const active = MODES.find((m) => m.id === mode)!;

  return (
    <div className="relative" ref={ref}>
      {/* Nút Trigger - Nằm gọn trên thanh input */}
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold transition-all hover:bg-[#e8e5e0]"
        style={{ color: "#1c1917", fontFamily: "system-ui, sans-serif" }}
      >
        <span style={{ color: "#c2640a" }}>{active.icon}</span>
        {active.label} <span style={{ color: "#78716c", fontWeight: "normal" }}>{active.sub}</span>
        <ChevronDown className={`w-3.5 h-3.5 text-[#78716c] transition-transform ml-1 ${open ? "rotate-180" : ""}`} />
      </button>

      {/* Menu thả xuống */}
      {open && (
        <div className="absolute bottom-full left-0 mb-2 w-72 rounded-2xl shadow-[0_8px_30px_rgb(0,0,0,0.12)] border overflow-hidden z-50 flex flex-col"
             style={{ background: "#ffffff", borderColor: "#e2dfd8", fontFamily: "system-ui, sans-serif" }}>
          
          <div className="px-4 py-3 border-b" style={{ borderColor: "#f0ede8", background: "#faf9f7" }}>
            <span className="text-xs font-bold tracking-wider uppercase" style={{ color: "#a09890" }}>Chọn Mô Hình AI</span>
          </div>

          <div className="p-1.5 flex flex-col gap-0.5">
            {MODES.map(m => {
              const isSelected = mode === m.id;
              return (
                <button
                  key={m.id}
                  onClick={() => { onChange(m.id); setOpen(false); }}
                  className="w-full flex items-center justify-between text-left px-3 py-2.5 rounded-xl transition-all border"
                  style={{ 
                    background: isSelected ? "#f5f4f0" : "transparent",
                    borderColor: isSelected ? "#e2dfd8" : "transparent"
                  }}
                >
                  <div className="flex flex-col gap-0.5 pr-2">
                    <span className="text-sm font-semibold" style={{ color: "#1c1917" }}>
                      {m.label} <span style={{ fontWeight: "normal" }}>{m.sub}</span>
                    </span>
                    <span className="text-xs leading-snug" style={{ color: "#78716c" }}>{m.desc}</span>
                  </div>
                  {/* Nút check tròn xanh/cam giống Gemini khi được chọn */}
                  {isSelected && (
                    <div className="w-5 h-5 rounded-full flex items-center justify-center shrink-0" style={{ background: "#1c1917" }}>
                       <Check className="w-3.5 h-3.5 text-white" strokeWidth={3} />
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main ───────────────────────────────────────────────────────
export default function VQAChatApp() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1, role: "bot",
      text: "Xin chào! Tôi là hệ thống phân tích võng mạc APTOS. Hãy tải lên ảnh fundus và chọn mô hình để bắt đầu phân tích.",
    },
  ]);
  const [mode, setMode]           = useState<Mode>("xai-eff");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [previewUrls, setPreviewUrls]     = useState<string[]>([]);
  const [note, setNote]           = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const fileRef   = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 180) + "px";
    }
  }, [note]);

  const addFiles = (files: File[]) => {
    const valid = files.filter(f => f.type.startsWith("image/"));
    if (!valid.length) return;
    setSelectedFiles(p => [...p, ...valid]);
    setPreviewUrls(p => [...p, ...valid.map(URL.createObjectURL)]);
  };

  const removeFile = (i: number) => {
    URL.revokeObjectURL(previewUrls[i]);
    setSelectedFiles(p => p.filter((_, idx) => idx !== i));
    setPreviewUrls(p => p.filter((_, idx) => idx !== i));
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const files: File[] = [];
    Array.from(e.clipboardData?.items ?? []).forEach(item => {
      if (item.type.startsWith("image/")) { const f = item.getAsFile(); if (f) files.push(f); }
    });
    if (files.length) { e.preventDefault(); addFiles(files); }
  };

  const handleSubmit = async () => {
    if (!selectedFiles.length && !note.trim()) return;

    const snapFiles    = [...selectedFiles];
    const snapPreviews = [...previewUrls];
    const snapMode     = mode;

    setMessages(p => [...p, {
      id: Date.now(), role: "user",
      text: note.trim() || undefined,
      previewUrls: snapPreviews.length ? snapPreviews : undefined,
    }]);

    setNote(""); setSelectedFiles([]); setPreviewUrls([]);
    if (fileRef.current) fileRef.current.value = "";
    setIsLoading(true);

    try {
      const form = new FormData();
      snapFiles.forEach(f => form.append("images", f));
      form.append("model_mode", snapMode);
      if (note.trim()) form.append("text_input", note);

      const res  = await fetch("http://localhost:8000/api/predict", { method: "POST", body: form });
      const data = await res.json();

      if (data.status !== "success") throw new Error(data.detail ?? "Lỗi server");

      setMessages(p => [...p, {
        id: Date.now() + 1, role: "bot",
        text: `Đã phân tích ${data.results.length} ảnh · Mô hình **${snapMode.toUpperCase()}**`,
        results: data.results,
      }]);
    } catch (err: any) {
      setMessages(p => [...p, { id: Date.now() + 1, role: "bot", text: `Lỗi: ${err.message}`, isError: true }]);
    } finally {
      setIsLoading(false);
    }
  };

  const canSend = !isLoading && (selectedFiles.length > 0 || note.trim().length > 0);
  const activeModeInfo = MODES.find(m => m.id === mode)!;

  return (
    <div className="flex flex-col h-screen"
         style={{ background: "#f5f4f0", fontFamily: "'Georgia', 'Times New Roman', serif" }}>

      {/* ── HEADER ── */}
      <header style={{ background: "#f5f4f0", borderBottom: "1px solid #e2dfd8" }}
              className="flex items-center justify-between px-6 py-3 shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-full flex items-center justify-center"
               style={{ background: "#c2640a" }}>
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div>
            <span className="text-sm font-semibold" style={{ color: "#1c1917", fontFamily: "Georgia, serif" }}>
              APTOS Research
            </span>
            <span className="text-xs ml-2" style={{ color: "#78716c" }}>Retinal Analysis</span>
          </div>
        </div>

        {/* Mode pill trên Header */}
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full border"
             style={{ borderColor: "#e2dfd8", background: "white" }}>
          <span style={{ color: "#c2640a" }}>{activeModeInfo.icon}</span>
          <span className="text-xs font-medium" style={{ color: "#44403c" }}>
            {activeModeInfo.label} <span style={{ color: "#a8a29e" }}>{activeModeInfo.sub}</span>
          </span>
        </div>
      </header>

      {/* ── CHAT ── */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 py-8 space-y-8">

          {messages.map(msg => (
            <div key={msg.id}>
              {msg.role === "user" ? (
                <div className="flex justify-end">
                  <div className="max-w-[75%] space-y-2">
                    {msg.previewUrls && (
                      <div className={`flex flex-wrap gap-2 justify-end`}>
                        {msg.previewUrls.map((url, i) => (
                          <img key={i} src={url} alt="upload"
                            className="rounded-2xl object-cover border border-[#e2dfd8]"
                            style={{ height: msg.previewUrls!.length === 1 ? 220 : 120,
                                     width:  msg.previewUrls!.length === 1 ? "auto" : 120 }} />
                        ))}
                      </div>
                    )}
                    {msg.text && (
                      <div className="px-4 py-3 rounded-2xl text-sm leading-relaxed"
                           style={{ background: "#1c1917", color: "#f5f4f0",
                                    fontFamily: "system-ui, sans-serif" }}>
                        {msg.text}
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="flex gap-3 items-start">
                  <div className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-0.5"
                       style={{ background: "#c2640a" }}>
                    <Sparkles className="w-4 h-4 text-white" />
                  </div>
                  <div className="flex-1 space-y-3 min-w-0">
                    {msg.text && (
                      <p className={`text-sm leading-relaxed ${msg.isError ? "text-red-600" : ""}`}
                         style={{ color: msg.isError ? "#dc2626" : "#1c1917",
                                  fontFamily: "system-ui, sans-serif" }}>
                        {msg.text}
                      </p>
                    )}
                    {msg.results && (
                      <div className="space-y-2">
                        {msg.results.map((r, i) => <ResultCard key={i} result={r} index={i} />)}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-3 items-center">
              <div className="w-8 h-8 rounded-full flex items-center justify-center"
                   style={{ background: "#c2640a" }}>
                <Sparkles className="w-4 h-4 text-white" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm" style={{ color: "#78716c", fontFamily: "system-ui" }}>
                  Đang phân tích
                </span>
                <span className="flex gap-1">
                  {[0,1,2].map(i => (
                    <span key={i} className="w-1.5 h-1.5 rounded-full animate-bounce inline-block"
                          style={{ background: "#c2640a", animationDelay: `${i * 0.15}s` }} />
                  ))}
                </span>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </main>

      {/* ── INPUT AREA ── */}
      <div className="shrink-0 px-4 pb-6 pt-2" style={{ background: "#f5f4f0" }}>
        <div className="max-w-3xl mx-auto">

          {/* SỬ DỤNG DROPDOWN COMPONENT TẠI ĐÂY */}
          <div className="flex mb-2 px-1">
            <ModelDropdown mode={mode} onChange={setMode} />
          </div>

          {/* Main input box */}
          <div className="rounded-3xl border overflow-hidden"
               style={{ borderColor: "#d6d3cc", background: "white",
                        boxShadow: "0 2px 8px rgba(0,0,0,0.06)" }}>

            {/* Image previews */}
            {previewUrls.length > 0 && (
              <div className="p-3 flex flex-wrap gap-2 border-b"
                   style={{ borderColor: "#f0ede8" }}>
                {previewUrls.map((url, i) => (
                  <div key={i} className="relative group">
                    <img src={url} className="h-16 w-16 object-cover rounded-xl border"
                         style={{ borderColor: "#e2dfd8" }} alt="preview" />
                    <button onClick={() => removeFile(i)}
                      className="absolute -top-1.5 -right-1.5 w-5 h-5 flex items-center justify-center rounded-full transition-opacity"
                      style={{ background: "#44403c" }}>
                      <X className="w-3 h-3 text-white" />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Textarea row */}
            <div className="flex items-end gap-2 px-4 py-3">
              <input type="file" multiple accept="image/*" className="hidden"
                     ref={fileRef} onChange={e => addFiles(Array.from(e.target.files ?? []))} />

              <button onClick={() => fileRef.current?.click()}
                      className="shrink-0 mb-0.5 transition-colors hover:text-[#1c1917]"
                      style={{ color: "#a8a29e" }}
                      title="Tải ảnh lên">
                <Plus className="w-5 h-5" />
              </button>

              <textarea
                ref={textareaRef}
                className="flex-1 bg-transparent outline-none resize-none text-sm leading-relaxed placeholder-[#a8a29e]"
                style={{ color: "#1c1917", fontFamily: "system-ui, sans-serif",
                         minHeight: "24px", maxHeight: "180px" }}
                placeholder="Nhập câu hỏi hoặc Ctrl+V để dán ảnh…"
                value={note}
                rows={1}
                onChange={e => setNote(e.target.value)}
                onPaste={handlePaste}
                onKeyDown={e => {
                  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
                }}
              />

              <button
                onClick={handleSubmit}
                disabled={!canSend}
                className="shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-all mb-0.5"
                style={{
                  background: canSend ? "#1c1917" : "#e2dfd8",
                  color: canSend ? "white" : "#a8a29e",
                }}>
                {isLoading
                  ? <Loader2 className="w-4 h-4 animate-spin" />
                  : <Send className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <p className="text-center text-[10px] mt-2" style={{ color: "#a8a29e", fontFamily: "system-ui" }}>
            APTOS Research · Phân tích võng mạc bằng AI · Kết quả chỉ mang tính tham khảo
          </p>
        </div>
      </div>
    </div>
  );
}