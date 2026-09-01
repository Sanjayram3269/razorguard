import { useState } from "react";

import { useQuery } from "@tanstack/react-query";

import {
  Building2,
  ChevronDown,
  ChevronRight,
  CreditCard,
  Fingerprint,
  Info,
  Layers,
  Link2,
  Network,
  ShieldAlert,
  ShieldCheck,
  Smartphone,
  Store,
  Users,
} from "lucide-react";

import clsx from "clsx";

import {
  fetchNetworkCluster,
  fetchNetworkTransaction,
  type NetworkTransaction,
  type RiskClusterResponse,
  type RiskClusterSignal,
} from "../lib/api";

/* ============================================================
   CONSTANTS
============================================================ */

const MAX_CONNECTED_ENTITIES = 8;

const SEVERITY_ORDER: Record<string, number> = {
  CRITICAL: 0,
  HIGH: 1,
  MEDIUM: 2,
  LOW: 3,
};

/* ============================================================
   TYPES
============================================================ */

interface EvidenceGraphV2Props {
  transactionId: string;
}

interface EntityNodeData {
  id: string;
  type: "transaction" | "account" | "device" | "merchant" | "connected";
  label: string;
  subtitle?: string;
  riskLevel?: "high" | "medium" | "low" | "none";
  accountType?: string;
}

/* ============================================================
   HELPERS
============================================================ */

function formatScore(value: number): string {
  if (!Number.isFinite(value)) {
    return "—";
  }

  return `${value.toFixed(1)}%`;
}

function severityClass(severity: string): string {
  return severity.toLowerCase();
}

function sortSignalsBySeverity(
  signals: RiskClusterSignal[],
): RiskClusterSignal[] {
  return [...signals].sort((a, b) => {
    const orderA = SEVERITY_ORDER[a.severity] ?? 99;
    const orderB = SEVERITY_ORDER[b.severity] ?? 99;

    return orderA - orderB;
  });
}

function buildEntityNodes(
  transaction: NetworkTransaction | undefined,
  cluster: RiskClusterResponse | undefined,
): EntityNodeData[] {
  if (!transaction) {
    return [];
  }

  const nodes: EntityNodeData[] = [];

  /* Transaction (center) */
  nodes.push({
    id: transaction.transaction_id,
    type: "transaction",
    label: transaction.transaction_id,
    subtitle: transaction.timestamp,
  });

  /* Account */
  const accountRisk = cluster?.accounts
    ? cluster.accounts.length > 1
      ? "high"
      : "low"
    : undefined;

  nodes.push({
    id: transaction.account_id,
    type: "account",
    label: transaction.account_id,
    subtitle: `${transaction.account_history_count} historical transactions`,
    riskLevel: accountRisk,
  });

  /* Device */
  const deviceIsShared =
    transaction.network_risk_signals.device_shared;

  nodes.push({
    id: transaction.device_id,
    type: "device",
    label: transaction.device_id,
    subtitle: transaction.network_risk_signals.new_device_for_account
      ? "New device for this account"
      : `${transaction.accounts_seen_on_device.length} accounts on device`,
    riskLevel: deviceIsShared ? "high" : "none",
  });

  /* Merchant */
  const merchantIsShared =
    transaction.network_risk_signals.merchant_shared;

  nodes.push({
    id: transaction.merchant_id,
    type: "merchant",
    label: transaction.merchant_id,
    subtitle: transaction.network_risk_signals.merchant_shared
      ? `Shared across ${transaction.accounts_seen_at_merchant.length} accounts`
      : `${transaction.accounts_seen_at_merchant.length} accounts at merchant`,
    riskLevel: merchantIsShared ? "medium" : "none",
  });

  /* Connected accounts (bounded) */
  if (cluster) {
    const otherAccounts = cluster.accounts
      .filter((acct) => acct !== transaction.account_id)
      .slice(0, MAX_CONNECTED_ENTITIES);

    otherAccounts.forEach((acct) => {
      nodes.push({
        id: acct,
        type: "connected",
        label: acct,
        subtitle: "Connected through coordination cluster",
        riskLevel: "medium",
        accountType: "connected",
      });
    });
  }

  return nodes;
}

/* ============================================================
   GRAPH NODE COMPONENT
============================================================ */

function GraphNode({
  node,
  isSelected,
  onClick,
}: {
  node: EntityNodeData;
  isSelected: boolean;
  onClick: () => void;
}) {
  const iconMap: Record<
    EntityNodeData["type"],
    typeof Building2
  > = {
    transaction: CreditCard,
    account: Building2,
    device: Smartphone,
    merchant: Store,
    connected: Users,
  };

  const Icon = iconMap[node.type];

  return (
    <button
      className={clsx(
        "ev2-node",
        `ev2-node-${node.type}`,
        node.riskLevel &&
          node.riskLevel !== "none" &&
          `ev2-risk-${node.riskLevel}`,
        isSelected && "ev2-selected",
      )}
      onClick={onClick}
      type="button"
    >
      <div className="ev2-node-icon">
        <Icon size={16} />
      </div>

      <span className="ev2-node-type">
        {node.type === "connected"
          ? "CONNECTED"
          : node.type.toUpperCase()}
      </span>

      <strong className="ev2-node-label">
        {node.label}
      </strong>

      {node.subtitle && (
        <small className="ev2-node-subtitle">
          {node.subtitle}
        </small>
      )}
    </button>
  );
}

/* ============================================================
   CONNECTOR COMPONENT
============================================================ */

function GraphConnector({
  label,
}: {
  label?: string;
}) {
  return (
    <div className="ev2-connector">
      <ChevronRight size={14} />

      {label && (
        <span className="ev2-connector-label">
          {label}
        </span>
      )}
    </div>
  );
}

/* ============================================================
   SIGNAL CARD
============================================================ */

function SignalCard({
  signal,
}: {
  signal: RiskClusterSignal;
}) {
  const iconMap: Record<string, typeof ShieldAlert> = {
    SHARED_DEVICE: Smartphone,
    SHARED_MERCHANT: Store,
    MULTI_ACCOUNT_CONNECTION: Users,
    TRANSACTION_CLUSTER: Layers,
    TEMPORAL_BURST: Fingerprint,
  };

  const Icon = iconMap[signal.type] ?? ShieldAlert;

  return (
    <div
      className={clsx(
        "ev2-signal",
        `ev2-signal-${severityClass(signal.severity)}`,
      )}
    >
      <div className="ev2-signal-icon">
        <Icon size={15} />
      </div>

      <div className="ev2-signal-content">
        <strong>{signal.evidence}</strong>

        <span>{signal.type.replace(/_/g, " ")}</span>
      </div>

      <span
        className={clsx(
          "ev2-signal-severity",
          `ev2-severity-${severityClass(signal.severity)}`,
        )}
      >
        {signal.severity}
      </span>
    </div>
  );
}

/* ============================================================
   ENTITY DETAIL PANEL
============================================================ */

function EntityDetailPanel({
  node,
  transaction,
  cluster,
}: {
  node: EntityNodeData;
  transaction: NetworkTransaction;
  cluster: RiskClusterResponse | undefined;
}) {
  if (node.type === "transaction") {
    return (
      <div className="ev2-detail">
        <div className="ev2-detail-header">
          <CreditCard size={16} />

          <span className="panel-eyebrow">
            TRANSACTION
          </span>
        </div>

        <h3>{node.label}</h3>

        <div className="ev2-detail-meta">
          <div>
            <span>Timestamp</span>

            <strong>{node.subtitle}</strong>
          </div>

          <div>
            <span>Related transactions</span>

            <strong>
              {transaction.related_transaction_count}
            </strong>
          </div>

          <div>
            <span>Account history</span>

            <strong>
              {transaction.account_history_count}
            </strong>
          </div>
        </div>

        <div className="ev2-detail-why">
          <Info size={13} />

          <span>
            This transaction is the center of the
            evidence graph. Connected entities are shown
            around it.
          </span>
        </div>
      </div>
    );
  }

  if (node.type === "account") {
    const isPartOfCluster =
      cluster &&
      cluster.accounts.includes(transaction.account_id);

    return (
      <div className="ev2-detail">
        <div className="ev2-detail-header">
          <Building2 size={16} />

          <span className="panel-eyebrow">
            ACCOUNT
          </span>
        </div>

        <h3>{node.label}</h3>

        <div className="ev2-detail-meta">
          <div>
            <span>Historical transactions</span>

            <strong>
              {transaction.account_history_count}
            </strong>
          </div>

          <div>
            <span>Accounts on shared device</span>

            <strong>
              {transaction.accounts_seen_on_device.length}
            </strong>
          </div>

          <div>
            <span>Accounts at merchant</span>

            <strong>
              {
                transaction.accounts_seen_at_merchant
                  .length
              }
            </strong>
          </div>
        </div>

        <div className="ev2-detail-why">
          {isPartOfCluster ? (
            <>
              <ShieldAlert size={13} />

              <span>
                This account is part of a coordinated
                risk cluster with{" "}
                {cluster.accounts.length} connected
                accounts.
                {cluster.cluster_type ===
                  "COORDINATED_NETWORK" &&
                  " The cluster shows coordinated network activity across multiple entities."}
              </span>
            </>
          ) : (
            <>
              <ShieldCheck size={13} />

              <span>
                This account has{" "}
                {transaction.account_history_count}{" "}
                historical transactions. No
                coordination cluster detected.
              </span>
            </>
          )}
        </div>
      </div>
    );
  }

  if (node.type === "device") {
    return (
      <div className="ev2-detail">
        <div className="ev2-detail-header">
          <Smartphone size={16} />

          <span className="panel-eyebrow">
            DEVICE
          </span>
        </div>

        <h3>{node.label}</h3>

        <div className="ev2-detail-meta">
          <div>
            <span>Accounts on device</span>

            <strong>
              {transaction.accounts_seen_on_device.length}
            </strong>
          </div>

          <div>
            <span>New for account</span>

            <strong>
              {transaction.network_risk_signals
                .new_device_for_account
                ? "Yes"
                : "No"}
            </strong>
          </div>

          <div>
            <span>Device shared</span>

            <strong>
              {transaction.network_risk_signals
                .device_shared
                ? "Yes"
                : "No"}
            </strong>
          </div>
        </div>

        <div className="ev2-detail-why">
          {transaction.network_risk_signals
            .device_shared ? (
            <>
              <ShieldAlert size={13} />

              <span>
                This device is shared across{" "}
                {
                  transaction.accounts_seen_on_device
                    .length
                }{" "}
                accounts. Shared devices are a strong
                indicator of coordinated activity.
              </span>
            </>
          ) : transaction.network_risk_signals
              .new_device_for_account ? (
            <>
              <Info size={13} />

              <span>
                This device has not been seen with this
                account before. New device usage can
                indicate account compromise.
              </span>
            </>
          ) : (
            <>
              <ShieldCheck size={13} />

              <span>
                This device is associated with this
                account's normal activity pattern.
              </span>
            </>
          )}
        </div>
      </div>
    );
  }

  if (node.type === "merchant") {
    return (
      <div className="ev2-detail">
        <div className="ev2-detail-header">
          <Store size={16} />

          <span className="panel-eyebrow">
            MERCHANT
          </span>
        </div>

        <h3>{node.label}</h3>

        <div className="ev2-detail-meta">
          <div>
            <span>Accounts at merchant</span>

            <strong>
              {
                transaction.accounts_seen_at_merchant
                  .length
              }
            </strong>
          </div>

          <div>
            <span>Merchant shared</span>

            <strong>
              {transaction.network_risk_signals
                .merchant_shared
                ? "Yes"
                : "No"}
            </strong>
          </div>

          <div>
            <span>New for account</span>

            <strong>
              {transaction.network_risk_signals
                .new_merchant_for_account
                ? "Yes"
                : "No"}
            </strong>
          </div>
        </div>

        <div className="ev2-detail-why">
          {transaction.network_risk_signals
            .merchant_shared ? (
            <>
              <ShieldAlert size={13} />

              <span>
                This merchant is shared across{" "}
                {
                  transaction.accounts_seen_at_merchant
                    .length
                }{" "}
                accounts. High merchant concentration
                can indicate coordinated purchasing
                patterns.
              </span>
            </>
          ) : (
            <>
              <ShieldCheck size={13} />

              <span>
                This merchant has normal account
                diversity. No unusual concentration
                detected.
              </span>
            </>
          )}
        </div>
      </div>
    );
  }

  /* Connected account from cluster */
  return (
    <div className="ev2-detail">
      <div className="ev2-detail-header">
        <Link2 size={16} />

        <span className="panel-eyebrow">
          CONNECTED ACCOUNT
        </span>
      </div>

      <h3>{node.label}</h3>

      <div className="ev2-detail-meta">
        {cluster && (
          <>
            <div>
              <span>Cluster type</span>

              <strong>
                {cluster.cluster_type.replace(
                  /_/g,
                  " ",
                )}
              </strong>
            </div>

            <div>
              <span>Cluster risk score</span>

              <strong>
                {formatScore(cluster.risk_score)}
              </strong>
            </div>

            <div>
              <span>Total connected accounts</span>

              <strong>
                {cluster.accounts.length}
              </strong>
            </div>
          </>
        )}
      </div>

      <div className="ev2-detail-why">
        <Link2 size={13} />

        <span>
          This account is connected to the primary
          account through the coordination cluster.
          {cluster &&
            ` ${cluster.evidence[0] ?? ""}`}
        </span>
      </div>
    </div>
  );
}

/* ============================================================
   COORDINATED RISK EVIDENCE
============================================================ */

function ClusterEvidencePanel({
  cluster,
}: {
  cluster: RiskClusterResponse;
}) {
  return (
    <div className="ev2-cluster">
      <div className="ev2-cluster-header">
        <div>
          <span className="panel-eyebrow">
            COORDINATED-RISK INTELLIGENCE
          </span>

          <h3>Cluster evidence</h3>
        </div>

        <div className="ev2-cluster-badge">
          <span
            className={clsx(
              "ev2-cluster-type",
              cluster.cluster_type ===
                "COORDINATED_NETWORK"
                ? "ev2-type-coordinated"
                : "ev2-type-connected",
            )}
          >
            {cluster.cluster_type.replace(/_/g, " ")}
          </span>

          <span className="ev2-cluster-score">
            {formatScore(cluster.risk_score)}
          </span>
        </div>
      </div>

      {/* Entity counts */}
      <div className="ev2-cluster-counts">
        <div className="ev2-cluster-count">
          <Users size={14} />

          <div>
            <strong>{cluster.accounts.length}</strong>

            <span>Accounts</span>
          </div>
        </div>

        <div className="ev2-cluster-count">
          <Smartphone size={14} />

          <div>
            <strong>{cluster.devices.length}</strong>

            <span>Devices</span>
          </div>
        </div>

        <div className="ev2-cluster-count">
          <Store size={14} />

          <div>
            <strong>{cluster.merchants.length}</strong>

            <span>Merchants</span>
          </div>
        </div>

        <div className="ev2-cluster-count">
          <Layers size={14} />

          <div>
            <strong>
              {cluster.transactions.length}
            </strong>

            <span>Transactions</span>
          </div>
        </div>
      </div>

      {/* Evidence statements */}
      {cluster.evidence.length > 0 && (
        <div className="ev2-evidence-list">
          <span className="panel-eyebrow">
            EVIDENCE
          </span>

          {cluster.evidence.map(
            (item, index) => (
              <div
                className="ev2-evidence-row"
                key={index}
              >
                <Info size={13} />

                <span>{item}</span>
              </div>
            ),
          )}
        </div>
      )}

      {/* Timeline preview */}
      {cluster.timeline.length > 0 && (
        <div className="ev2-timeline">
          <span className="panel-eyebrow">
            TRANSACTION TIMELINE
          </span>

          <div className="ev2-timeline-list">
            {cluster.timeline
              .slice(0, 6)
              .map((item, index) => (
                <div
                  className="ev2-timeline-item"
                  key={index}
                >
                  <div className="ev2-timeline-dot" />

                  <div className="ev2-timeline-content">
                    <strong>
                      {item.transaction_id}
                    </strong>

                    <span>
                      {item.account_id} ·{" "}
                      {item.device_id} ·{" "}
                      {new Date(
                        item.timestamp,
                      ).toLocaleString([], {
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                  </div>
                </div>
              ))}

            {cluster.timeline.length > 6 && (
              <div className="ev2-timeline-more">
                +{cluster.timeline.length - 6}{" "}
                more transactions
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ============================================================
   MAIN COMPONENT
============================================================ */

export default function EvidenceGraphV2({
  transactionId,
}: EvidenceGraphV2Props) {
  const [selectedEntity, setSelectedEntity] =
    useState<EntityNodeData | null>(null);

  const [signalsExpanded, setSignalsExpanded] =
    useState(true);

  /* ---- Fetch network transaction ---- */
  const {
    data: transaction,
    isLoading: txLoading,
    isError: txError,
  } = useQuery({
    queryKey: ["network", "transaction", transactionId],
    queryFn: () =>
      fetchNetworkTransaction(transactionId),
    enabled: Boolean(transactionId),
  });

  /* ---- Fetch cluster intelligence ---- */
  const {
    data: cluster,
    isLoading: clusterLoading,
  } = useQuery({
    queryKey: ["network", "cluster", transactionId],
    queryFn: () => fetchNetworkCluster(transactionId),
    enabled: Boolean(transactionId),
  });

  const isLoading = txLoading || clusterLoading;
  const isError = txError;

  /* ---- Loading state ---- */
  if (isLoading) {
    return (
      <div className="ev2-container">
        <div className="ev2-loading">
          <Network size={18} />

          <span>Loading evidence graph...</span>
        </div>
      </div>
    );
  }

  /* ---- Error state ---- */
  if (isError || !transaction) {
    return (
      <div className="ev2-container">
        <div className="ev2-empty">
          <ShieldAlert size={22} />

          <div>
            <strong>Evidence graph unavailable</strong>

            <span>
              Network intelligence could not be loaded
              for this transaction.
            </span>
          </div>
        </div>
      </div>
    );
  }

  /* ---- Build entity nodes ---- */
  const entityNodes = buildEntityNodes(
    transaction,
    cluster,
  );

  /* ---- Sort signals by severity ---- */
  const sortedSignals = cluster
    ? sortSignalsBySeverity(cluster.signals)
    : [];

  /* ---- Primary entity (transaction) ---- */
  const transactionNode = entityNodes.find(
    (n) => n.type === "transaction",
  );

  /* ---- Direct relationships (account, device, merchant) ---- */
  const directEntities = entityNodes.filter(
    (n) =>
      n.type === "account" ||
      n.type === "device" ||
      n.type === "merchant",
  );

  /* ---- Connected entities from cluster ---- */
  const connectedEntities = entityNodes.filter(
    (n) => n.type === "connected",
  );

  return (
    <div className="ev2-container">
      {/* ===== HEADER ===== */}
      <div className="ev2-header">
        <div>
          <span className="panel-eyebrow">
            EVIDENCE GRAPH V2
          </span>

          <h2>Transaction relationship map</h2>

          <p>
            Interactive investigation view of entity
            relationships and coordinated-risk evidence.
          </p>
        </div>

        <div className="ev2-header-badges">
          {cluster && (
            <span
              className={clsx(
                "ev2-cluster-type",
                cluster.cluster_type ===
                  "COORDINATED_NETWORK"
                  ? "ev2-type-coordinated"
                  : "ev2-type-connected",
              )}
            >
              {cluster.cluster_type.replace(/_/g, " ")}
            </span>
          )}
        </div>
      </div>

      {/* ===== GRAPH CANVAS ===== */}
      <div className="ev2-canvas">
        {/* Transaction node */}
        {transactionNode && (
          <>
            <GraphNode
              node={transactionNode}
              isSelected={
                selectedEntity?.id ===
                transactionNode.id
              }
              onClick={() =>
                setSelectedEntity(transactionNode)
              }
            />

            <GraphConnector label="involves" />
          </>
        )}

        {/* Direct entity nodes */}
        <div className="ev2-direct-entities">
          {directEntities.map((node) => (
            <GraphNode
              key={node.id}
              node={node}
              isSelected={
                selectedEntity?.id === node.id
              }
              onClick={() =>
                setSelectedEntity(node)
              }
            />
          ))}
        </div>

        {/* Connected entities from cluster */}
        {connectedEntities.length > 0 && (
          <>
            <GraphConnector label="coordinated" />

            <div className="ev2-connected-entities">
              {connectedEntities.map((node) => (
                <GraphNode
                  key={node.id}
                  node={node}
                  isSelected={
                    selectedEntity?.id === node.id
                  }
                  onClick={() =>
                    setSelectedEntity(node)
                  }
                />
              ))}

              {cluster &&
                cluster.accounts.length >
                  MAX_CONNECTED_ENTITIES + 1 && (
                  <div className="ev2-more-entities">
                    +
                    {cluster.accounts.length -
                      MAX_CONNECTED_ENTITIES -
                      1}{" "}
                    more accounts
                  </div>
                )}
            </div>
          </>
        )}
      </div>

      {/* ===== TWO-COLUMN: Detail + Signals ===== */}
      <div className="ev2-body">
        {/* Entity Detail Panel */}
        <div className="ev2-detail-panel">
          {selectedEntity ? (
            <EntityDetailPanel
              node={selectedEntity}
              transaction={transaction}
              cluster={cluster}
            />
          ) : (
            <div className="ev2-detail ev2-detail-empty">
              <div className="ev2-detail-header">
                <Info size={16} />

                <span className="panel-eyebrow">
                  SELECT AN ENTITY
                </span>
              </div>

              <p>
                Click any entity node above to view
                its relationship details and
                investigative context.
              </p>
            </div>
          )}
        </div>

        {/* Risk Signals Panel */}
        <div className="ev2-signals-panel">
          <button
            className="ev2-signals-toggle"
            onClick={() =>
              setSignalsExpanded(
                (prev) => !prev,
              )
            }
            type="button"
          >
            <div>
              <span className="panel-eyebrow">
                NETWORK RISK SIGNALS
              </span>

              <h3>
                {sortedSignals.length} signal
                {sortedSignals.length !== 1
                  ? "s"
                  : ""}{" "}
                detected
              </h3>
            </div>

            {sortedSignals.length > 0 && (
              <ChevronDown
                size={16}
                className={clsx(
                  !signalsExpanded &&
                    "ev2-chevron-collapsed",
                )}
              />
            )}
          </button>

          {signalsExpanded &&
            sortedSignals.length > 0 && (
              <div className="ev2-signal-list">
                {sortedSignals.map(
                  (signal, index) => (
                    <SignalCard
                      key={`${signal.type}-${index}`}
                      signal={signal}
                    />
                  ),
                )}
              </div>
            )}

          {signalsExpanded &&
            sortedSignals.length === 0 && (
              <div className="ev2-signal-empty">
                <ShieldCheck size={18} />

                <span>
                  No coordinated-risk signals
                  detected for this transaction.
                </span>
              </div>
            )}
        </div>
      </div>

      {/* ===== CLUSTER EVIDENCE ===== */}
      {cluster && (
        <ClusterEvidencePanel cluster={cluster} />
      )}
    </div>
  );
}
