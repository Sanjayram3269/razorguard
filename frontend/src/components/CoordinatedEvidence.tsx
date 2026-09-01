import { useState } from "react";

import {
  ChevronDown,
  Fingerprint,
  Info,
  Layers,
  Network,
  ShieldAlert,
  ShieldCheck,
  Smartphone,
  Store,
} from "lucide-react";

import clsx from "clsx";

import type { PrioritizedEvidenceData } from "../lib/api";

/* ============================================================
   PROPS
============================================================ */

interface CoordinatedEvidenceProps {
  evidence: PrioritizedEvidenceData[];
  summary: Record<string, number>;
}

/* ============================================================
   COMPONENT
============================================================ */

export default function CoordinatedEvidence({
  evidence,
  summary,
}: CoordinatedEvidenceProps) {
  const [expandedTiers, setExpandedTiers] = useState<
    Record<string, boolean>
  >({
    PRIMARY: true,
    SUPPORTING: true,
    CONTEXTUAL: false,
  });

  const toggleTier = (tier: string) => {
    setExpandedTiers((prev) => ({
      ...prev,
      [tier]: !prev[tier],
    }));
  };

  const primaryItems = evidence.filter(
    (e) => e.tier === "PRIMARY",
  );

  const supportingItems = evidence.filter(
    (e) => e.tier === "SUPPORTING",
  );

  const contextualItems = evidence.filter(
    (e) => e.tier === "CONTEXTUAL",
  );

  if (evidence.length === 0) {
    return (
      <div className="cev-empty">
        <ShieldCheck size={18} />

        <span>
          No coordinated-risk evidence detected for
          this case.
        </span>
      </div>
    );
  }

  return (
    <div className="cev-container">
      {/* Summary badges */}
      <div className="cev-summary">
        {summary.PRIMARY > 0 && (
          <span className="cev-badge cev-badge-primary">
            {summary.PRIMARY} primary
          </span>
        )}

        {summary.SUPPORTING > 0 && (
          <span className="cev-badge cev-badge-supporting">
            {summary.SUPPORTING} supporting
          </span>
        )}

        {summary.CONTEXTUAL > 0 && (
          <span className="cev-badge cev-badge-contextual">
            {summary.CONTEXTUAL} contextual
          </span>
        )}
      </div>

      {/* PRIMARY tier */}
      <EvidenceTier
        tier="PRIMARY"
        label="Primary evidence"
        description="Strongest reasons for this case"
        items={primaryItems}
        isExpanded={expandedTiers.PRIMARY}
        onToggle={() => toggleTier("PRIMARY")}
      />

      {/* SUPPORTING tier */}
      <EvidenceTier
        tier="SUPPORTING"
        label="Supporting evidence"
        description="Corroborating signals"
        items={supportingItems}
        isExpanded={expandedTiers.SUPPORTING}
        onToggle={() => toggleTier("SUPPORTING")}
      />

      {/* CONTEXTUAL tier */}
      <EvidenceTier
        tier="CONTEXTUAL"
        label="Contextual evidence"
        description="Background information"
        items={contextualItems}
        isExpanded={expandedTiers.CONTEXTUAL}
        onToggle={() => toggleTier("CONTEXTUAL")}
      />
    </div>
  );
}

/* ============================================================
   EVIDENCE TIER
============================================================ */

function EvidenceTier({
  tier,
  label,
  description,
  items,
  isExpanded,
  onToggle,
}: {
  tier: string;
  label: string;
  description: string;
  items: PrioritizedEvidenceData[];
  isExpanded: boolean;
  onToggle: () => void;
}) {
  if (items.length === 0) {
    return null;
  }

  return (
    <div className={`cev-tier cev-tier-${tier.toLowerCase()}`}>
      <button
        className="cev-tier-header"
        onClick={onToggle}
        type="button"
      >
        <div className="cev-tier-info">
          <span className={`cev-tier-dot cev-dot-${tier.toLowerCase()}`} />

          <div>
            <strong>{label}</strong>

            <span>{description}</span>
          </div>
        </div>

        <div className="cev-tier-count">
          <span>{items.length}</span>

          <ChevronDown
            size={14}
            className={clsx(
              !isExpanded && "cev-chevron-collapsed",
            )}
          />
        </div>
      </button>

      {isExpanded && (
        <div className="cev-tier-items">
          {items.map((item, index) => (
            <EvidenceCard
              item={item}
              key={`${item.category}-${item.title}-${index}`}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/* ============================================================
   EVIDENCE CARD
============================================================ */

function EvidenceCard({
  item,
}: {
  item: PrioritizedEvidenceData;
}) {
  const icon = categoryIcon(item.category);
  const Icon = icon;

  return (
    <div
      className={clsx(
        "cev-card",
        `cev-severity-${item.severity.toLowerCase()}`,
      )}
    >
      <div className="cev-card-header">
        <div className="cev-card-icon">
          <Icon size={14} />
        </div>

        <div className="cev-card-title-area">
          <strong>{item.title}</strong>

          <span className="cev-card-category">
            {item.category.replace(/_/g, " ")}
          </span>
        </div>

        <span
          className={`cev-severity-badge cev-sev-${item.severity.toLowerCase()}`}
        >
          {item.severity}
        </span>
      </div>

      <p className="cev-card-explanation">
        {item.explanation}
      </p>

      <div className="cev-card-relevance">
        <Info size={11} />

        <span>{item.investigative_relevance}</span>
      </div>

      {item.observed_value && (
        <div className="cev-card-observed">
          Observed: {item.observed_value}
        </div>
      )}

      {item.supporting_entities.length > 0 && (
        <div className="cev-card-entities">
          {item.supporting_entities
            .slice(0, 4)
            .map((entity, i) => (
              <span className="cev-entity-tag" key={i}>
                {entity}
              </span>
            ))}

          {item.supporting_entities.length > 4 && (
            <span className="cev-entity-more">
              +{item.supporting_entities.length - 4} more
            </span>
          )}
        </div>
      )}
    </div>
  );
}

/* ============================================================
   HELPERS
============================================================ */

function categoryIcon(
  category: string,
): typeof ShieldAlert {
  switch (category) {
    case "CONVERGENCE":
      return Layers;

    case "CLUSTER":
      return Fingerprint;

    case "NETWORK":
      return Network;

    case "BEHAVIORAL":
      return Smartphone;

    case "TRANSACTION":
      return Store;

    default:
      return ShieldAlert;
  }
}


