import { useQuery } from "@tanstack/react-query";

import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock3,
  FileSearch,
  Info,
  Loader2,
  UserCheck,
  XCircle,
} from "lucide-react";

import { fetchCaseAudit, type AuditEvent } from "../lib/api";

/* ============================================================
   PROPS
============================================================ */

interface InvestigationTimelineProps {
  caseId: string;
}

/* ============================================================
   HELPERS
============================================================ */

function formatTimestamp(value: string): string {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function actionIcon(action: string): typeof Activity {
  switch (action) {
    case "CASE_CREATED":
      return FileSearch;

    case "CASE_ASSIGNED":
      return UserCheck;

    case "STATUS_CHANGED":
      return AlertTriangle;

    case "CASE_RESOLVED":
      return CheckCircle2;

    case "CASE_ESCALATED":
      return AlertTriangle;

    default:
      return Activity;
  }
}

function actionLabel(action: string): string {
  return action
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/^./, (char) => char.toUpperCase());
}

function isSystemActor(actor: string): boolean {
  const lower = actor.toLowerCase();
  return (
    lower === "system" ||
    lower === "api" ||
    lower === "automated"
  );
}

/* ============================================================
   COMPONENT
============================================================ */

export default function InvestigationTimeline({
  caseId,
}: InvestigationTimelineProps) {
  const {
    data,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["case", caseId, "audit"],
    queryFn: () => fetchCaseAudit(caseId),
    enabled: Boolean(caseId),
  });

  const events = data?.events ?? [];

  /* ---- Loading ---- */
  if (isLoading) {
    return (
      <div className="timeline-loading">
        <Loader2 size={16} className="spin" />

        <span>Loading audit history...</span>
      </div>
    );
  }

  /* ---- Error ---- */
  if (isError) {
    return (
      <div className="timeline-empty">
        <XCircle size={18} />

        <span>Unable to load audit history.</span>
      </div>
    );
  }

  /* ---- Empty ---- */
  if (events.length === 0) {
    return (
      <div className="timeline-empty">
        <Clock3 size={18} />

        <span>No audit events recorded yet.</span>
      </div>
    );
  }

  /* ---- Render ---- */
  return (
    <div className="timeline-container">
      {events.map(
        (event: AuditEvent, index: number) => {
          const Icon = actionIcon(event.action);
          const isFirst = index === 0;
          const isLast = index === events.length - 1;
          const systemActor = isSystemActor(
            event.actor,
          );

          return (
            <div
              className={`timeline-event ${
                isFirst ? "timeline-first" : ""
              } ${isLast ? "timeline-last" : ""}`}
              key={`${event.case_id}-${event.timestamp}-${event.action}`}
            >
              {/* Vertical line */}
              <div className="timeline-track">
                <div
                  className={`timeline-dot ${
                    isFirst ? "timeline-dot-active" : ""
                  } ${systemActor ? "timeline-dot-system" : ""}`}
                />

                {!isLast && (
                  <div className="timeline-line" />
                )}
              </div>

              {/* Content */}
              <div className="timeline-content">
                <div className="timeline-header">
                  <div className="timeline-icon">
                    <Icon size={13} />
                  </div>

                  <span className="timeline-action">
                    {actionLabel(event.action)}
                  </span>

                  <time className="timeline-time">
                    {formatTimestamp(event.timestamp)}
                  </time>
                </div>

                <div className="timeline-meta">
                  <span
                    className={`timeline-actor ${
                      systemActor
                        ? "timeline-actor-system"
                        : "timeline-actor-human"
                    }`}
                  >
                    {systemActor ? (
                      <>
                        <Info size={10} />

                        System
                      </>
                    ) : (
                      <>
                        <UserCheck size={10} />

                        {event.actor}
                      </>
                    )}
                  </span>

                  {event.from_status &&
                    event.to_status && (
                      <span className="timeline-transition">
                        {event.from_status.replace(
                          "_",
                          " ",
                        )}

                        {" → "}

                        {event.to_status.replace(
                          "_",
                          " ",
                        )}
                      </span>
                    )}
                </div>

                {event.details && (
                  <div className="timeline-details">
                    {event.details}
                  </div>
                )}
              </div>
            </div>
          );
        },
      )}
    </div>
  );
}
