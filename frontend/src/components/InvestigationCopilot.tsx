import { useState } from "react";

import { useMutation, useQuery } from "@tanstack/react-query";

import {
  Bot,
  CheckCircle2,
  Info,
  Loader2,
  Send,
  Sparkles,
  XCircle,
} from "lucide-react";

import {
  askCopilot,
  fetchCopilotStatus,
  type CopilotResponseData,
} from "../lib/api";

/* ============================================================
   PROPS
============================================================ */

interface InvestigationCopilotProps {
  caseId: string;
}

/* ============================================================
   SUGGESTED QUESTIONS
============================================================ */

const SUGGESTED_QUESTIONS = [
  "Why was this case flagged?",
  "What is the strongest evidence?",
  "Summarize this case for me.",
  "What relationships should I review?",
  "What should I investigate next?",
  "Explain the coordinated-risk cluster.",
];

/* ============================================================
   COMPONENT
============================================================ */

export default function InvestigationCopilot({
  caseId,
}: InvestigationCopilotProps) {
  const [question, setQuestion] = useState("");
  const [showResponse, setShowResponse] = useState(false);

  /* ---- Status ---- */
  const { data: status } = useQuery({
    queryKey: ["copilot", "status"],
    queryFn: fetchCopilotStatus,
  });

  /* ---- Ask mutation ---- */
  const askMutation = useMutation({
    mutationFn: (q: string) => askCopilot(caseId, q),
    onSuccess: () => {
      setShowResponse(true);
    },
  });

  const handleSubmit = () => {
    const q = question.trim();

    if (!q) {
      return;
    }

    askMutation.mutate(q);
  };

  const handleSuggested = (q: string) => {
    setQuestion(q);
    askMutation.mutate(q);
  };

  const isUnavailable = status && !status.available;

  return (
    <div className="copilot-container">
      {/* Header */}
      <div className="copilot-header">
        <div className="copilot-header-left">
          <div className="copilot-icon">
            <Bot size={16} />
          </div>

          <div>
            <strong>Investigation Copilot</strong>

            <span className="copilot-subtitle">
              {isUnavailable
                ? "Unavailable — using deterministic fallback"
                : "AI-assisted investigation analysis"}
            </span>
          </div>
        </div>

        {isUnavailable && (
          <span className="copilot-status-badge copilot-status-off">
            <XCircle size={11} />

            Offline
          </span>
        )}

        {status?.available && (
          <span className="copilot-status-badge copilot-status-on">
            <CheckCircle2 size={11} />

            Ready
          </span>
        )}
      </div>

      {/* Input */}
      <div className="copilot-input-area">
        <div className="copilot-input-row">
          <input
            className="copilot-input"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                handleSubmit();
              }
            }}
            placeholder="Ask about this case..."
            disabled={askMutation.isPending}
          />

          <button
            className="copilot-send"
            onClick={handleSubmit}
            disabled={!question.trim() || askMutation.isPending}
            type="button"
          >
            {askMutation.isPending ? (
              <Loader2 size={14} className="spin" />
            ) : (
              <Send size={14} />
            )}
          </button>
        </div>

        {/* Suggested questions */}
        {!showResponse && !askMutation.isPending && (
          <div className="copilot-suggestions">
            {SUGGESTED_QUESTIONS.map((q) => (
              <button
                className="copilot-suggestion"
                key={q}
                onClick={() => handleSuggested(q)}
                type="button"
              >
                <Sparkles size={11} />

                {q}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Loading */}
      {askMutation.isPending && (
        <div className="copilot-loading">
          <Loader2 size={16} className="spin" />

          <span>Analyzing evidence...</span>
        </div>
      )}

      {/* Error */}
      {askMutation.isError && (
        <div className="copilot-error">
          <XCircle size={16} />

          <span>
            Unable to get copilot response. The
            deterministic evidence panels remain
            available.
          </span>
        </div>
      )}

      {/* Response */}
      {askMutation.isSuccess && askMutation.data && (
        <CopilotResponseDisplay
          response={askMutation.data}
        />
      )}

      {/* Disclosure */}
      <div className="copilot-disclosure">
        <Info size={10} />

        <span>
          {isUnavailable
            ? "Copilot unavailable. Showing deterministic evidence summaries. Verify against RazorGuard evidence."
            : "AI-generated investigation assistance. Verify recommendations against RazorGuard evidence."}
        </span>
      </div>
    </div>
  );
}

/* ============================================================
   RESPONSE DISPLAY
============================================================ */

function CopilotResponseDisplay({
  response,
}: {
  response: CopilotResponseData;
}) {
  return (
    <div className="copilot-response">
      {/* Answer */}
      <div className="copilot-answer">
        <div className="copilot-answer-header">
          <Bot size={13} />

          <span className="copilot-grounding-badge">
            {response.grounding.includes("VERIFIED")
              ? "Verified Evidence"
              : "AI Interpretation"}
          </span>
        </div>

        <div className="copilot-answer-text">
          {response.answer.split("\n").map(
            (line, i) => (
              <p key={i}>{line}</p>
            ),
          )}
        </div>
      </div>

      {/* Key Evidence */}
      {response.key_evidence.length > 0 && (
        <div className="copilot-key-evidence">
          <span className="copilot-section-label">
            Key Evidence
          </span>

          <div className="copilot-evidence-tags">
            {response.key_evidence.map(
              (item, i) => (
                <span
                  className="copilot-evidence-tag"
                  key={i}
                >
                  {item}
                </span>
              ),
            )}
          </div>
        </div>
      )}

      {/* Recommended Focus */}
      {response.recommended_focus && (
        <div className="copilot-focus">
          <span className="copilot-section-label">
            Recommended Focus
          </span>

          <span className="copilot-focus-text">
            {response.recommended_focus}
          </span>
        </div>
      )}

      {/* Interpretation */}
      {response.interpretation && (
        <div className="copilot-interpretation">
          <Info size={11} />

          <span>{response.interpretation}</span>
        </div>
      )}
    </div>
  );
}
