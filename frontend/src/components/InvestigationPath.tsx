import {
  ArrowRight,
  BarChart3,
  CheckCircle2,
  FileSearch,
  Fingerprint,
  Info,
  Network,
  Search,
  ShieldAlert,
  Smartphone,
  Store,
  Users,
} from "lucide-react";

import type { InvestigationStepData } from "../lib/api";

import { useNavigate } from "react-router-dom";

/* ============================================================
   PROPS
============================================================ */

interface InvestigationPathProps {
  steps: InvestigationStepData[];
}

/* ============================================================
   COMPONENT
============================================================ */

export default function InvestigationPath({
  steps,
}: InvestigationPathProps) {
  const navigate = useNavigate();

  if (steps.length === 0) {
    return (
      <div className="path-empty">
        <CheckCircle2 size={18} />

        <span>
          No specific investigation steps recommended.
          All available evidence has been reviewed.
        </span>
      </div>
    );
  }

  return (
    <div className="path-container">
      {steps.map((step) => {
        const Icon = iconForTarget(
          step.target_entity,
        );

        return (
          <button
            className={`path-step path-priority-${step.priority}`}
            key={`${step.priority}-${step.title}`}
            onClick={() => handleNavigation(
              step.navigation_target,
              navigate,
            )}
            type="button"
          >
            {/* Priority indicator */}
            <div className="path-rank">
              <span>{step.priority}</span>
            </div>

            {/* Content */}
            <div className="path-content">
              <div className="path-header">
                <div className="path-icon">
                  <Icon size={14} />
                </div>

                <strong className="path-title">
                  {step.title}
                </strong>
              </div>

              <p className="path-reason">
                {step.reason}
              </p>

              {step.supporting_evidence.length >
                0 && (
                <div className="path-evidence">
                  {step.supporting_evidence.map(
                    (item, i) => (
                      <span
                        className="path-evidence-tag"
                        key={i}
                      >
                        <Info size={10} />

                        {item}
                      </span>
                    ),
                  )}
                </div>
              )}
            </div>

            {/* Navigation arrow */}
            <ArrowRight
              size={14}
              className="path-arrow"
            />
          </button>
        );
      })}
    </div>
  );
}

/* ============================================================
   HELPERS
============================================================ */

function iconForTarget(
  target: string,
): typeof ShieldAlert {
  switch (target) {
    case "device":
      return Smartphone;

    case "merchant":
      return Store;

    case "account":
      return Users;

    case "cluster":
      return Network;

    case "transactions":
      return Fingerprint;

    case "network":
      return Network;

    case "analytics":
      return BarChart3;

    case "case":
      return FileSearch;

    default:
      return Search;
  }
}

function handleNavigation(
  target: string,
  navigate: ReturnType<typeof useNavigate>,
): void {
  switch (target) {
    case "network":
      navigate("/network");

      break;

    case "analytics":
      navigate("/analytics");

      break;

    case "cases":
      navigate("/cases");

      break;

    default:
      break;
  }
}
