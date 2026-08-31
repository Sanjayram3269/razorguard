import { useState } from "react";
import EvidenceGraph from "./components/EvidenceGraph";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Bell,
  CheckCircle2,
  CircleHelp,
  Clock3,
  ChevronDown,
  FileSearch,
  LayoutDashboard,
  Loader2,
  Network,
  RefreshCw,
  Settings,
  Search,
  ShieldAlert,
  Users,
  XCircle,
} from "lucide-react";

import {
  BrowserRouter,
  NavLink,
  Navigate,
  Route,
  Routes,
  useNavigate,
} from "react-router-dom";

import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import {
  assignCase,
  fetchAnalyticsOverview,
  fetchCase,
  fetchCasesFiltered,
  fetchDashboardActivity,
  fetchDashboardDistribution,
  fetchDashboardQueue,
  fetchDashboardSummary,
  fetchNetworkSummary,
  fetchNetworkTransaction,
  transitionCase,
  type AnalyticsDistributionItem,
  type AnalyticsMetricResponse,
  type DashboardActivityItem,
  type DashboardDistributionItem,
  type DashboardQueueItem,
} from "./lib/api";
import "./App.css";

/* ============================================================
   NAVIGATION
============================================================ */

const navigation = [
  {
    label: "Command Center",
    icon: LayoutDashboard,
    path: "/",
  },
  {
    label: "Case Queue",
    icon: FileSearch,
    path: "/cases",
  },
  {
    label: "Network Intelligence",
    icon: Network,
    path: "/network",
  },
  {
    label: "Risk Analytics",
    icon: BarChart3,
    path: "/analytics",
  },
];

const secondaryNavigation = [
  {
    label: "Investigators",
    icon: Users,
    path: "/investigators",
  },
  {
    label: "Activity",
    icon: Activity,
    path: "/activity",
  },
  {
    label: "Settings",
    icon: Settings,
    path: "/settings",
  },
];

/* ============================================================
   HELPERS
============================================================ */

function formatScore(score: number) {
  if (!Number.isFinite(score)) {
    return "—";
  }

  return `${score.toFixed(1)}%`;
}

function formatDate(value: string) {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return "—";
  }

  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function priorityClass(priority: string) {
  return priority.toLowerCase();
}

function statusClass(status: string) {
  return status.toLowerCase().replace("_", "-");
}

/* ============================================================
   STAT CARD
============================================================ */

function StatCard({
  label,
  value,
  detail,
  icon: Icon,
  tone = "neutral",
}: {
  label: string;
  value: string | number;
  detail: string;
  icon: typeof ShieldAlert;
  tone?: "neutral" | "danger" | "warning" | "success";
}) {
  return (
    <div className={`stat-card ${tone}`}>
      <div className="stat-top">
        <span>{label}</span>

        <div className="stat-icon">
          <Icon size={16} />
        </div>
      </div>

      <div className="stat-value">
        {value}
      </div>

      <div className="stat-detail">
        {detail}
      </div>
    </div>
  );
}

/* ============================================================
   RISK DISTRIBUTION
============================================================ */

function RiskDistribution({
  items,
  total,
}: {
  items: DashboardDistributionItem[];
  total: number;
}) {
  const rows = [
    {
      label: "Critical",
      key: "CRITICAL",
    },
    {
      label: "High",
      key: "HIGH",
    },
    {
      label: "Medium",
      key: "MEDIUM",
    },
    {
      label: "Low",
      key: "LOW",
    },
  ];

  return (
    <div className="panel distribution-panel">
      <div className="panel-heading">
        <div>
          <span className="panel-eyebrow">
            RISK DISTRIBUTION
          </span>

          <h2>Case severity</h2>
        </div>

        <div className="panel-meta">
          {total} total
        </div>
      </div>

      <div className="distribution-list">
        {rows.map((row) => {
          const item = items.find(
            (entry) =>
              entry.label.toUpperCase() === row.key,
          );

          const count = item?.count ?? 0;
          const percentage =
            item?.percentage ?? 0;

          return (
            <div
              className="distribution-row"
              key={row.key}
            >
              <div className="distribution-label">
                <span
                  className={`risk-dot ${row.key.toLowerCase()}`}
                />

                <span>
                  {row.label}
                </span>

                <strong>
                  {count}
                </strong>
              </div>

              <div className="distribution-track">
                <div
                  className={`distribution-bar ${row.key.toLowerCase()}`}
                  style={{
                    width: `${Math.max(
                      percentage,
                      count > 0 ? 3 : 0,
                    )}%`,
                  }}
                />
              </div>

              <span className="distribution-percent">
                {percentage.toFixed(0)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ============================================================
   PRIORITY QUEUE
============================================================ */

function PriorityQueue({
  items,
}: {
  items: DashboardQueueItem[];
}) {
  const navigate = useNavigate();

  return (
    <div className="panel queue-panel">
      <div className="panel-heading">
        <div>
          <span className="panel-eyebrow">
            INVESTIGATOR QUEUE
          </span>

          <h2>Priority cases</h2>
        </div>

        <button
          className="text-button"
          onClick={() =>
            navigate("/cases")
          }
        >
          View all
        </button>
      </div>

      {items.length === 0 ? (
        <div className="empty-state compact">
          <CheckCircle2 size={22} />

          <div>
            <strong>
              No active cases
            </strong>

            <span>
              The investigation queue is
              currently empty.
            </span>
          </div>
        </div>
      ) : (
        <div className="case-list">
          {items.map((item) => (
            <button
              className="case-row"
              key={item.case_id}
              onClick={() =>
                navigate(
                  `/cases/${encodeURIComponent(
                    item.case_id,
                  )}`,
                )
              }
            >
              <div className="case-row-main">
                <span className="case-id">
                  {item.case_id}
                </span>

                <span className="case-reason">
                  {item.primary_reason ||
                    "Risk investigation"}
                </span>
              </div>

              <div className="case-row-risk">
                <span
                  className={`priority-badge ${priorityClass(
                    item.priority,
                  )}`}
                >
                  {item.priority}
                </span>

                <strong>
                  {formatScore(
                    item.risk_score,
                  )}
                </strong>

                <span className="case-arrow">
                  →
                </span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ============================================================
   ACTIVITY PANEL
============================================================ */

function ActivityPanel({
  items,
}: {
  items: DashboardActivityItem[];
}) {
  const recent = items.slice(0, 5);

  return (
    <div className="panel activity-panel">
      <div className="panel-heading">
        <div>
          <span className="panel-eyebrow">
            SYSTEM ACTIVITY
          </span>

          <h2>Recent activity</h2>
        </div>

        <Activity size={17} />
      </div>

      {recent.length === 0 ? (
        <div className="empty-state compact">
          <Clock3 size={21} />

          <div>
            <strong>
              No recent activity
            </strong>

            <span>
              Activity will appear as
              cases change.
            </span>
          </div>
        </div>
      ) : (
        <div className="activity-list">
          {recent.map((item) => (
            <div
              className="activity-row"
              key={`${item.case_id}-${item.timestamp}`}
            >
              <div className="activity-icon">
                {item.action ===
                "CASE_RESOLVED" ? (
                  <CheckCircle2 size={15} />
                ) : item.action ===
                  "CASE_ESCALATED" ? (
                  <AlertTriangle size={15} />
                ) : (
                  <Activity size={15} />
                )}
              </div>

              <div className="activity-copy">
                <strong>
                  {item.case_id}
                </strong>

                <span>
                  {item.action
                    .replaceAll("_", " ")
                    .toLowerCase()
                    .replace(
                      /^./,
                      (char) =>
                        char.toUpperCase(),
                    )}
                </span>
              </div>

              <time>
                {formatDate(
                  item.timestamp,
                )}
              </time>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ============================================================
   DASHBOARD
============================================================ */

function Dashboard() {
  const {
    data: summary,
    isLoading: summaryLoading,
    isError: summaryError,
    refetch: refetchSummary,
    isFetching: summaryFetching,
  } = useQuery({
    queryKey: ["dashboard", "summary"],
    queryFn: fetchDashboardSummary,
  });

  const {
    data: distribution,
    isLoading: distributionLoading,
    isError: distributionError,
    refetch: refetchDistribution,
  } = useQuery({
    queryKey: ["dashboard", "distribution"],
    queryFn: fetchDashboardDistribution,
  });

  const {
    data: activity,
    isLoading: activityLoading,
    isError: activityError,
    refetch: refetchActivity,
  } = useQuery({
    queryKey: ["dashboard", "activity"],
    queryFn: () =>
      fetchDashboardActivity(10),
  });

  const {
    data: queue,
    isLoading: queueLoading,
    isError: queueError,
    refetch: refetchQueue,
  } = useQuery({
    queryKey: ["dashboard", "queue"],
    queryFn: () =>
      fetchDashboardQueue(10),
  });

  const isLoading =
    summaryLoading ||
    distributionLoading ||
    activityLoading ||
    queueLoading;

  const isError =
    summaryError ||
    distributionError ||
    activityError ||
    queueError;

  const isFetching =
    summaryFetching;

  const refetchAll = async () => {
    await Promise.all([
      refetchSummary(),
      refetchDistribution(),
      refetchActivity(),
      refetchQueue(),
    ]);
  };

  const metrics = {
    open:
      summary?.open_cases ?? 0,

    critical:
      summary?.critical_cases ?? 0,

    high:
      summary?.high_cases ?? 0,

    average:
      summary?.average_risk_score ?? 0,
  };

  return (
    <div className="dashboard">
      <div className="page-header">
        <div>
          <span className="eyebrow">
            OVERVIEW
          </span>

          <h1>
            Risk Command Center
          </h1>

          <p>
            Live operational view of
            transaction risk and
            investigator workload.
          </p>
        </div>

        <button
          className="refresh-button"
          onClick={() =>
            refetchAll()
          }
          disabled={isFetching}
        >
          <RefreshCw
            size={15}
            className={
              isFetching
                ? "spin"
                : ""
            }
          />

          {isFetching
            ? "Refreshing"
            : "Refresh"}
        </button>
      </div>

      {isError && (
        <div className="error-banner">
          <XCircle size={17} />

          <div>
            <strong>
              Risk data unavailable
            </strong>

            <span>
              Unable to reach the
              RazorGuard API. Make sure
              the FastAPI server is
              running on port 8000.
            </span>
          </div>

          <button
            onClick={() =>
              refetchAll()
            }
          >
            Retry
          </button>
        </div>
      )}

      {isLoading ? (
        <DashboardSkeleton />
      ) : (
        <>
          <div className="stats-grid">
            <StatCard
              label="OPEN CASES"
              value={metrics.open}
              detail="Requires investigator attention"
              icon={FileSearch}
            />

            <StatCard
              label="CRITICAL"
              value={metrics.critical}
              detail="Highest-priority investigations"
              icon={ShieldAlert}
              tone="danger"
            />

            <StatCard
              label="HIGH RISK"
              value={metrics.high}
              detail="Elevated fraud indicators"
              icon={AlertTriangle}
              tone="warning"
            />

            <StatCard
              label="AVG RISK SCORE"
              value={formatScore(
                metrics.average,
              )}
              detail="Across available cases"
              icon={BarChart3}
              tone="success"
            />
          </div>

          <div className="dashboard-grid">
            <RiskDistribution
              items={
                distribution?.items ??
                []
              }
              total={
                distribution?.total ??
                0
              }
            />

            <ActivityPanel
              items={
                activity?.items ??
                []
              }
            />

            <PriorityQueue
              items={
                queue?.items ??
                []
              }
            />
          </div>
        </>
      )}
    </div>
  );
}

/* ============================================================
   SKELETON
============================================================ */

function DashboardSkeleton() {
  return (
    <div className="skeleton-layout">
      <div className="stats-grid">
        {[1, 2, 3, 4].map(
          (item) => (
            <div
              className="skeleton stat-skeleton"
              key={item}
            />
          ),
        )}
      </div>

      <div className="dashboard-grid">
        <div className="skeleton large-skeleton" />
        <div className="skeleton large-skeleton" />
        <div className="skeleton queue-skeleton" />
      </div>
    </div>
  );
}

/* ============================================================
   CASE QUEUE PAGE
============================================================ */

function Cases() {
  const {
    data,
    isLoading,
    isError,
    refetch,
  } = useQuery({
    queryKey: ["cases", "queue"],
    queryFn: () =>
      fetchCasesFiltered({
        page: 1,
        page_size: 50,
        sort_by: "risk_score",
        sort_order: "desc",
      }),
  });

  const [search, setSearch] =
    useState("");

  const [priority, setPriority] =
    useState("ALL");

  const cases = data?.cases ?? [];

  const filteredCases =
    cases.filter((item) => {
      const searchValue =
        search.toLowerCase();

      const searchMatch =
        search.trim() === "" ||
        item.case_id
          .toLowerCase()
          .includes(searchValue) ||
        item.transaction_id
          .toLowerCase()
          .includes(searchValue) ||
        item.primary_reason
          .toLowerCase()
          .includes(searchValue);

      const priorityMatch =
        priority === "ALL" ||
        item.priority.toUpperCase() ===
          priority;

      return (
        searchMatch &&
        priorityMatch
      );
    });

  const navigate =
    useNavigate();

  return (
    <div className="dashboard">
      <div className="page-header">
        <div>
          <span className="eyebrow">
            INVESTIGATIONS
          </span>

          <h1>Case Queue</h1>

          <p>
            Review, prioritize, and open
            investigator cases.
          </p>
        </div>

        <button
          className="refresh-button"
          onClick={() => refetch()}
        >
          <RefreshCw size={15} />
          Refresh
        </button>
      </div>

      <div className="queue-toolbar">
        <div className="search-box">
          <Search size={16} />

          <input
            value={search}
            onChange={(event) =>
              setSearch(
                event.target.value,
              )
            }
            placeholder="Search cases, transactions, reasons..."
          />
        </div>

        <div className="filter-group">
          {[
            "ALL",
            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW",
          ].map((item) => (
            <button
              key={item}
              className={
                priority === item
                  ? "filter-button active"
                  : "filter-button"
              }
              onClick={() =>
                setPriority(item)
              }
            >
              {item}
            </button>
          ))}
        </div>
      </div>

      <div className="panel full-panel">
        {isLoading ? (
          <div className="loading-state">
            <Loader2
              size={20}
              className="spin"
            />

            Loading cases...
          </div>
        ) : isError ? (
          <div className="empty-state">
            <XCircle size={28} />

            <strong>
              Unable to load cases
            </strong>

            <span>
              Check that the RazorGuard
              API is running.
            </span>
          </div>
        ) : filteredCases.length ===
          0 ? (
          <div className="empty-state">
            <CheckCircle2 size={28} />

            <strong>
              No matching cases
            </strong>

            <span>
              Try changing your filters or
              search query.
            </span>
          </div>
        ) : (
          <div className="table-wrap">
            <table className="case-table">
              <thead>
                <tr>
                  <th>CASE</th>
                  <th>PRIORITY</th>
                  <th>RISK</th>
                  <th>STATUS</th>
                  <th>ASSIGNED</th>
                  <th>UPDATED</th>
                </tr>
              </thead>

              <tbody>
                {filteredCases.map(
                  (item) => (
                    <tr
                      key={item.case_id}
                      onClick={() =>
                        navigate(
                          `/cases/${encodeURIComponent(
                            item.case_id,
                          )}`,
                        )
                      }
                    >
                      <td>
                        <div className="table-case">
                          <strong>
                            {item.case_id}
                          </strong>

                          <span>
                            {item.primary_reason ||
                              "Risk investigation"}
                          </span>
                        </div>
                      </td>

                      <td>
                        <span
                          className={`priority-badge ${priorityClass(
                            item.priority,
                          )}`}
                        >
                          {item.priority}
                        </span>
                      </td>

                      <td>
                        <strong>
                          {formatScore(
                            item.risk_score,
                          )}
                        </strong>
                      </td>

                      <td>
                        <span
                          className={`status-badge ${statusClass(
                            item.status,
                          )}`}
                        >
                          {item.status.replace(
                            "_",
                            " ",
                          )}
                        </span>
                      </td>

                      <td>
                        {item.assigned_to ??
                          "Unassigned"}
                      </td>

                      <td>
                        {formatDate(
                          item.updated_at,
                        )}
                      </td>
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

/* ============================================================
   NETWORK INTELLIGENCE
============================================================ */

function NetworkIntelligence() {
  const [transactionId, setTransactionId] =
    useState("");

  const [searchedTransaction, setSearchedTransaction] =
    useState("");

  const {
    data: summary,
    isLoading: summaryLoading,
    isError: summaryError,
    refetch: refetchSummary,
  } = useQuery({
    queryKey: ["network", "summary"],
    queryFn: fetchNetworkSummary,
  });

  const {
    data: transaction,
    isLoading: transactionLoading,
    isError: transactionError,
    refetch: refetchTransaction,
  } = useQuery({
    queryKey: [
      "network",
      "transaction",
      searchedTransaction,
    ],
    queryFn: () =>
      fetchNetworkTransaction(
        searchedTransaction,
      ),
    enabled: Boolean(searchedTransaction),
  });

  const handleInvestigate = () => {
    const value = transactionId.trim();

    if (!value) {
      return;
    }

    setSearchedTransaction(value);
  };

  const handleRefresh = async () => {
    await Promise.all([
      refetchSummary(),
      searchedTransaction
        ? refetchTransaction()
        : Promise.resolve(),
    ]);
  };

  const signals =
    transaction?.network_risk_signals;

  return (
    <div className="dashboard">
      <div className="page-header">
        <div>
          <span className="eyebrow">
            NETWORK INTELLIGENCE
          </span>

          <h1>
            Relationship Intelligence
          </h1>

          <p>
            Explore account, device, merchant,
            and transaction relationships.
          </p>
        </div>

        <button
          className="refresh-button"
          onClick={handleRefresh}
          disabled={
            summaryLoading ||
            transactionLoading
          }
        >
          <RefreshCw
            size={15}
            className={
              summaryLoading ||
              transactionLoading
                ? "spin"
                : ""
            }
          />

          {summaryLoading ||
          transactionLoading
            ? "Refreshing"
            : "Refresh"}
        </button>
      </div>

      {summaryError && (
        <div className="error-banner">
          <XCircle size={17} />

          <div>
            <strong>
              Network data unavailable
            </strong>

            <span>
              Unable to load network intelligence
              from the RazorGuard API.
            </span>
          </div>

          <button
            onClick={() =>
              refetchSummary()
            }
          >
            Retry
          </button>
        </div>
      )}

      <div className="stats-grid">
        <StatCard
          label="ACCOUNTS"
          value={summary?.accounts ?? 0}
          detail="Known transaction accounts"
          icon={Users}
        />

        <StatCard
          label="DEVICES"
          value={summary?.devices ?? 0}
          detail="Observed transaction devices"
          icon={Network}
        />

        <StatCard
          label="MERCHANTS"
          value={summary?.merchants ?? 0}
          detail="Connected merchant entities"
          icon={ShieldAlert}
        />

        <StatCard
          label="RELATIONSHIPS"
          value={
            (summary?.account_device_edges ?? 0) +
            (summary?.account_merchant_edges ?? 0) +
            (summary?.device_merchant_edges ?? 0)
          }
          detail="Observed entity connections"
          icon={Activity}
        />
      </div>

      <div className="dashboard-grid">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <span className="panel-eyebrow">
                NETWORK TOPOLOGY
              </span>

              <h2>
                Relationship coverage
              </h2>
            </div>

            <Network size={17} />
          </div>

          <div className="distribution-list">
            <NetworkRelationshipRow
              label="Account ↔ Device"
              count={
                summary?.account_device_edges ??
                0
              }
            />

            <NetworkRelationshipRow
              label="Account ↔ Merchant"
              count={
                summary?.account_merchant_edges ??
                0
              }
            />

            <NetworkRelationshipRow
              label="Device ↔ Merchant"
              count={
                summary?.device_merchant_edges ??
                0
              }
            />
          </div>
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <span className="panel-eyebrow">
                TRANSACTION INVESTIGATION
              </span>

              <h2>
                Trace a transaction
              </h2>
            </div>

            <Search size={17} />
          </div>

          <div className="network-search">
            <div className="search-box">
              <Search size={16} />

              <input
                value={transactionId}
                onChange={(event) =>
                  setTransactionId(
                    event.target.value,
                  )
                }
                onKeyDown={(event) => {
                  if (
                    event.key === "Enter"
                  ) {
                    handleInvestigate();
                  }
                }}
                placeholder="Enter transaction ID..."
              />
            </div>

            <button
              className="refresh-button"
              onClick={handleInvestigate}
              disabled={
                !transactionId.trim() ||
                transactionLoading
              }
            >
              {transactionLoading
                ? "Investigating..."
                : "Investigate"}
            </button>
          </div>

          {!searchedTransaction ? (
            <div className="empty-state compact">
              <FileSearch size={22} />

              <div>
                <strong>
                  No transaction selected
                </strong>

                <span>
                  Enter a transaction ID to
                  inspect its network relationships.
                </span>
              </div>
            </div>
          ) : transactionError ? (
            <div className="empty-state compact">
              <XCircle size={22} />

              <div>
                <strong>
                  Transaction unavailable
                </strong>

                <span>
                  No network intelligence could
                  be loaded for {searchedTransaction}.
                </span>
              </div>
            </div>
          ) : transaction ? (
            <div className="network-result">
              <div className="network-entities">
                <NetworkEntity
                  label="ACCOUNT"
                  value={transaction.account_id}
                />

                <NetworkEntity
                  label="DEVICE"
                  value={transaction.device_id}
                />

                <NetworkEntity
                  label="MERCHANT"
                  value={transaction.merchant_id}
                />
              </div>

              <div className="network-metrics">
                <div>
                  <span>
                    Account history
                  </span>

                  <strong>
                    {transaction.account_history_count}
                  </strong>
                </div>

                <div>
                  <span>
                    Related transactions
                  </span>

                  <strong>
                    {transaction.related_transaction_count}
                  </strong>
                </div>

                <div>
                  <span>
                    Accounts at merchant
                  </span>

                  <strong>
                    {
                      transaction
                        .accounts_seen_at_merchant
                        .length
                    }
                  </strong>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </div>

    {transaction && (
      <EvidenceGraph
        transaction={transaction}
      />
    )}

    {transaction && (
      <div className="panel">
        <div className="panel-heading">
            <div>
              <span className="panel-eyebrow">
                NETWORK RISK SIGNALS
              </span>

              <h2>
                Relationship signals
              </h2>
            </div>

            <ShieldAlert size={17} />
          </div>

          <div className="signal-grid network-signal-grid">
            <NetworkSignal
              label="Shared device"
              active={
                signals?.device_shared ??
                false
              }
            />

            <NetworkSignal
              label="Shared merchant"
              active={
                signals?.merchant_shared ??
                false
              }
            />

            <NetworkSignal
              label="New device for account"
              active={
                signals?.new_device_for_account ??
                false
              }
            />

            <NetworkSignal
              label="New merchant for account"
              active={
                signals?.new_merchant_for_account ??
                false
              }
            />
          </div>
        </div>
      )}
    </div>
  );
}

function NetworkRelationshipRow({
  label,
  count,
}: {
  label: string;
  count: number;
}) {
  return (
    <div className="distribution-row">
      <div className="distribution-label">
        <span className="risk-dot medium" />

        <span>
          {label}
        </span>

        <strong>
          {count.toLocaleString()}
        </strong>
      </div>

      <div className="distribution-track">
        <div
          className="distribution-bar medium"
          style={{
            width: `${Math.max(
              Math.min(count / 500, 100),
              3,
            )}%`,
          }}
        />
      </div>
    </div>
  );
}

function NetworkEntity({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="network-entity">
      <span>
        {label}
      </span>

      <strong>
        {value}
      </strong>
    </div>
  );
}

function NetworkSignal({
  label,
  active,
}: {
  label: string;
  active: boolean;
}) {
  return (
    <div
      className={`network-signal ${
        active ? "active" : ""
      }`}
    >
      {active ? (
        <CheckCircle2 size={18} />
      ) : (
        <XCircle size={18} />
      )}

      <div>
        <strong>
          {label}
        </strong>

        <span>
          {active
            ? "Signal detected"
            : "No signal detected"}
        </span>
      </div>
    </div>
  );
}

/* ============================================================
   RISK ANALYTICS
============================================================ */

function RiskAnalytics() {
  const {
    data,
    isLoading,
    isError,
    isFetching,
    refetch,
  } = useQuery<AnalyticsMetricResponse>({
    queryKey: ["analytics", "overview"],
    queryFn: fetchAnalyticsOverview,
  });

  const formatPercentage = (value: number) => {
    if (!Number.isFinite(value)) {
      return "0.00%";
    }

    return `${value.toFixed(2)}%`;
  };

  const formatCount = (value: number) => {
    if (!Number.isFinite(value)) {
      return "0";
    }

    return value.toLocaleString();
  };

  const formatLabel = (value: string) =>
    value
      .replaceAll("_", " ")
      .toLowerCase()
      .replace(
        /^./,
        (character) => character.toUpperCase(),
      );

  const renderDistribution = (
    items: AnalyticsDistributionItem[],
  ) => {
    if (items.length === 0) {
      return (
        <div className="empty-state compact">
          <CheckCircle2 size={22} />

          <div>
            <strong>
              No distribution data
            </strong>

            <span>
              Analytics data is currently unavailable.
            </span>
          </div>
        </div>
      );
    }

    return (
      <div className="analytics-list">
        {items.map((item) => {
          const width = Math.min(
  Math.max(item.percentage, 0),
  100,
);

          return (
            <div
              className="analytics-row"
              key={item.label}
            >
              <div className="analytics-row-header">
                <span>
                  {formatLabel(item.label)}
                </span>

                <strong>
                  {formatCount(item.count)}
                </strong>
              </div>

              <div className="analytics-track">
                <div
                  className="analytics-bar"
                  style={{
                    width: `${Math.max(
                      width,
                      item.count > 0 ? 1 : 0,
                    )}%`,
                  }}
                />
              </div>

              <span className="analytics-percentage">
                {formatPercentage(
                  item.percentage,
                )}
              </span>
            </div>
          );
        })}
      </div>
    );
  };

  const refetchAnalytics = async () => {
    await refetch();
  };

  if (isLoading) {
    return (
      <div className="loading-state page-empty">
        <Loader2
          size={20}
          className="spin"
        />

        Loading risk analytics...
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="empty-state page-empty">
        <XCircle size={30} />

        <strong>
          Risk analytics unavailable
        </strong>

        <span>
          Unable to load analytics from the
          RazorGuard API.
        </span>

        <button
          className="refresh-button"
          onClick={refetchAnalytics}
          disabled={isFetching}
        >
          <RefreshCw
            size={15}
            className={
              isFetching ? "spin" : ""
            }
          />

          {isFetching
            ? "Retrying..."
            : "Retry"}
        </button>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className="page-header">
        <div>
          <span className="eyebrow">
            RISK ANALYTICS
          </span>

          <h1>
            Risk Intelligence
          </h1>

          <p>
            Analyze risk distribution, model
            behavior, decisions, and
            investigation trends.
          </p>
        </div>

        <button
          className="refresh-button"
          onClick={refetchAnalytics}
          disabled={isFetching}
        >
          <RefreshCw
            size={15}
            className={
              isFetching ? "spin" : ""
            }
          />

          {isFetching
            ? "Refreshing"
            : "Refresh"}
        </button>
      </div>

      {/* ========================================================
         CORE METRICS
      ======================================================== */}

      <div className="stats-grid">
        <StatCard
          label="TOTAL CASES"
          value={formatCount(
            data.total_cases,
          )}
          detail="Cases analyzed by the risk engine"
          icon={FileSearch}
        />

        <StatCard
          label="AVERAGE RISK"
          value={data.average_risk_score.toFixed(
            2,
          )}
          detail="Mean risk score across cases"
          icon={Activity}
          tone="warning"
        />

        <StatCard
          label="MEDIAN RISK"
          value={data.median_risk_score.toFixed(
            2,
          )}
          detail="Median observed risk score"
          icon={BarChart3}
        />

        <StatCard
          label="MAXIMUM RISK"
          value={data.maximum_risk_score.toFixed(
            2,
          )}
          detail="Highest observed risk score"
          icon={ShieldAlert}
          tone="danger"
        />
      </div>

      {/* ========================================================
         MODEL + NETWORK
      ======================================================== */}

      <div className="dashboard-grid">
        <div className="panel analytics-model-panel">
          <div className="panel-heading">
          <div>
      <span className="panel-eyebrow">
        MODEL INTELLIGENCE
      </span>

              <h2>
                Model behavior
              </h2>
            </div>

            <BarChart3 size={17} />
          </div>

          <div className="analytics-metric-grid">
            <div className="analytics-metric">
              <span>
                Average model probability
              </span>

              <strong>
                {formatPercentage(
                  data.average_model_probability *
                    100,
                )}
              </strong>
            </div>

            <div className="analytics-metric">
              <span>
                Average network score
              </span>

              <strong>
                {data.average_network_score.toFixed(
                  2,
                )}
              </strong>
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <span className="panel-eyebrow">
                PRIORITY DISTRIBUTION
              </span>

              <h2>
                Case priority
              </h2>
            </div>

            <AlertTriangle size={17} />
          </div>

          {renderDistribution(
            data.priority_distribution,
          )}
        </div>
      </div>

      {/* ========================================================
         RISK + DECISIONS
      ======================================================== */}

      <div className="dashboard-grid">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <span className="panel-eyebrow">
                RISK LEVEL DISTRIBUTION
              </span>

              <h2>
                Risk severity
              </h2>
            </div>

            <ShieldAlert size={17} />
          </div>

          {renderDistribution(
            data.risk_level_distribution,
          )}
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <span className="panel-eyebrow">
                DECISION DISTRIBUTION
              </span>

              <h2>
                Engine decisions
              </h2>
            </div>

            <CheckCircle2 size={17} />
          </div>

          {renderDistribution(
            data.decision_distribution,
          )}
        </div>
      </div>

      {/* ========================================================
         STATUS + TOP REASONS
      ======================================================== */}

      <div className="dashboard-grid">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <span className="panel-eyebrow">
                INVESTIGATION STATUS
              </span>

              <h2>
                Case lifecycle
              </h2>
            </div>

            <Activity size={17} />
          </div>

          {renderDistribution(
            data.status_distribution,
          )}
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <span className="panel-eyebrow">
                PRIMARY RISK REASONS
              </span>

              <h2>
                Dominant signals
              </h2>
            </div>

            <Search size={17} />
          </div>

          {renderDistribution(
            data.top_reasons,
          )}
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   INVESTIGATION
============================================================ */

function Investigation({
  caseId,
}: {
  caseId?: string;
}) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [investigator, setInvestigator] = useState("");

  /* ============================================================
     CASE DATA
  ============================================================ */

  const {
    data: caseData,
    isLoading,
    isError,
    refetch: refetchCase,
  } = useQuery({
    queryKey: ["case", caseId],
    queryFn: () => fetchCase(caseId!),
    enabled: Boolean(caseId),
  });

  /* ============================================================
     ASSIGN INVESTIGATOR

     IMPORTANT:
     This hook must stay before every conditional return.
  ============================================================ */

  const assignMutation = useMutation({
    mutationFn: () =>
      assignCase(caseId!, {
        investigator: investigator.trim(),
        actor: "security-analyst",
      }),

    onSuccess: async () => {
      setInvestigator("");

      await queryClient.invalidateQueries({
        queryKey: ["case", caseId],
      });

      await queryClient.invalidateQueries({
        queryKey: ["cases"],
      });

      await queryClient.invalidateQueries({
        queryKey: ["dashboard"],
      });
    },
  });

  /* ============================================================
     STATUS TRANSITION

     IMPORTANT:
     This hook must also stay before every conditional return.
  ============================================================ */

  const statusMutation = useMutation({
    mutationFn: ({
      status,
      details,
    }: {
      status: string;
      details: string;
    }) =>
      transitionCase(caseId!, {
        status,
        actor: "security-analyst",
        details,
      }),

    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["case", caseId],
      });

      await queryClient.invalidateQueries({
        queryKey: ["cases"],
      });

      await queryClient.invalidateQueries({
        queryKey: ["dashboard"],
      });

      await queryClient.invalidateQueries({
        queryKey: ["analytics", "overview"],
      });

      await queryClient.invalidateQueries({
        queryKey: ["dashboard", "activity"],
      });
    },
  });

  /* ============================================================
     NO CASE ID
  ============================================================ */

  if (!caseId) {
    return (
      <div className="empty-state page-empty">
        <FileSearch size={30} />

        <strong>
          Investigation workspace
        </strong>

        <span>
          Select a case from the queue.
        </span>
      </div>
    );
  }

  /* ============================================================
     LOADING
  ============================================================ */

  if (isLoading) {
    return (
      <div className="loading-state page-empty">
        <Loader2
          size={20}
          className="spin"
        />

        Loading investigation...
      </div>
    );
  }

  /* ============================================================
     ERROR
  ============================================================ */

  if (isError || !caseData) {
    return (
      <div className="empty-state page-empty">
        <XCircle size={30} />

        <strong>
          Case unavailable
        </strong>

        <span>
          This investigation could not be
          loaded.
        </span>

        <div className="case-actions">
          <button
            className="refresh-button"
            onClick={() => refetchCase()}
          >
            <RefreshCw size={15} />
            Retry
          </button>

          <button
            className="refresh-button"
            onClick={() => navigate("/cases")}
          >
            Return to queue
          </button>
        </div>
      </div>
    );
  }

  /* ============================================================
     INVESTIGATION VIEW
  ============================================================ */

  return (
    <div className="dashboard">
      <button
        className="back-button"
        onClick={() => navigate("/cases")}
      >
        ← Back to cases
      </button>

      <div className="page-header">
        <div>
          <span className="eyebrow">
            INVESTIGATION
          </span>

          <h1>
            {caseData.case_id}
          </h1>

          <p>
            Transaction {caseData.transaction_id}
          </p>
        </div>

        <span
          className={`priority-badge large ${priorityClass(
            caseData.priority,
          )}`}
        >
          {caseData.priority}
        </span>
      </div>

      <div className="investigation-grid">
        <div className="panel investigation-score">
          <span className="panel-eyebrow">
            RISK SCORE
          </span>

          <div className="big-score">
            {formatScore(caseData.risk_score)}
          </div>

          <span
            className={`status-badge ${statusClass(
              caseData.status,
            )}`}
          >
            {caseData.status.replace("_", " ")}
          </span>
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <span className="panel-eyebrow">
                RISK SIGNALS
              </span>

              <h2>
                Decision context
              </h2>
            </div>

            <ShieldAlert size={17} />
          </div>

          <div className="signal-grid">
            <div>
              <span>
                Model probability
              </span>

              <strong>
                {formatScore(
                  caseData.model_probability * 100,
                )}
              </strong>
            </div>

            <div>
              <span>
                Network score
              </span>

              <strong>
                {formatScore(caseData.network_score)}
              </strong>
            </div>

            <div>
              <span>
                Decision
              </span>

              <strong>
                {caseData.decision}
              </strong>
            </div>

            <div>
              <span>
                Assigned investigator
              </span>

              <strong>
                {caseData.assigned_to ?? "Unassigned"}
              </strong>

              <div className="case-action">
                <input
                  type="text"
                  value={investigator}
                  onChange={(event) =>
                    setInvestigator(event.target.value)
                  }
                  placeholder="Investigator name"
                  disabled={assignMutation.isPending}
                />

                <button
                  className="refresh-button"
                  onClick={() => assignMutation.mutate()}
                  disabled={
                    assignMutation.isPending ||
                    !investigator.trim()
                  }
                >
                  {assignMutation.isPending
                    ? "Assigning..."
                    : "Assign"}
                </button>
              </div>

              {assignMutation.isError && (
                <span className="action-error">
                  Unable to assign investigator.
                </span>
              )}

              {assignMutation.isSuccess && (
                <span className="action-success">
                  Investigator assigned successfully.
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="panel investigation-panel">
        <div className="panel-heading">
          <div>
            <span className="panel-eyebrow">
              CASE ACTIONS
            </span>

            <h2>
              Investigation workflow
            </h2>
          </div>

          <ShieldAlert size={17} />
        </div>

        <div className="case-actions">
          {caseData.status === "OPEN" && (
            <button
              className="refresh-button"
              onClick={() =>
                statusMutation.mutate({
                  status: "IN_REVIEW",
                  details:
                    "Investigation review started by security analyst.",
                })
              }
              disabled={statusMutation.isPending}
            >
              {statusMutation.isPending
                ? "Updating..."
                : "Start Review"}
            </button>
          )}

          {caseData.status === "IN_REVIEW" && (
            <>
              <button
                className="refresh-button"
                onClick={() =>
                  statusMutation.mutate({
                    status: "ESCALATED",
                    details:
                      "Case escalated for additional investigation.",
                  })
                }
                disabled={statusMutation.isPending}
              >
                {statusMutation.isPending
                  ? "Updating..."
                  : "Escalate"}
              </button>

              <button
                className="refresh-button"
                onClick={() =>
                  statusMutation.mutate({
                    status: "RESOLVED",
                    details:
                      "Investigation completed and case resolved.",
                  })
                }
                disabled={statusMutation.isPending}
              >
                {statusMutation.isPending
                  ? "Updating..."
                  : "Resolve Case"}
              </button>
            </>
          )}

          {caseData.status === "ESCALATED" && (
            <button
              className="refresh-button"
              onClick={() =>
                statusMutation.mutate({
                  status: "RESOLVED",
                  details:
                    "Escalated investigation completed and case resolved.",
                })
              }
              disabled={statusMutation.isPending}
            >
              {statusMutation.isPending
                ? "Updating..."
                : "Resolve Case"}
            </button>
          )}

          {caseData.status === "RESOLVED" && (
            <div className="action-success">
              <CheckCircle2 size={16} />
              Case resolved. No further action required.
            </div>
          )}

          {statusMutation.isError && (
            <span className="action-error">
              Unable to update case status. Please try again.
            </span>
          )}

          {statusMutation.isSuccess && (
            <span className="action-success">
              Case status updated successfully.
            </span>
          )}
        </div>
      </div>

      <div className="panel investigation-panel">
        <div className="panel-heading">
          <div>
            <span className="panel-eyebrow">
              PRIMARY REASON
            </span>

            <h2>
              {caseData.primary_reason ||
                "Risk indicators detected"}
            </h2>
          </div>

          <ShieldAlert size={17} />
        </div>

        <div className="evidence-block">
          <span className="panel-eyebrow">
            EVIDENCE
          </span>

          <p>
            {caseData.evidence_text ||
              "No evidence text recorded."}
          </p>
        </div>

        <div className="evidence-block">
          <span className="panel-eyebrow">
            INVESTIGATION NARRATIVE
          </span>

          <p>
            {caseData.investigation_narrative ||
              "No investigation narrative recorded."}
          </p>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   INVESTIGATORS
============================================================ */

function Investigators() {
  const navigate = useNavigate();

  const {
    data: casesData,
    isLoading,
    isError,
    refetch,
  } = useQuery({
    queryKey: ["cases", "investigators"],
    queryFn: () =>
      fetchCasesFiltered({
        page: 1,
        page_size: 50,
        sort_by: "risk_score",
        sort_order: "desc",
      }),
  });

  const cases = casesData?.cases ?? [];
  const assigned = cases.filter((item) => Boolean(item.assigned_to));
  const unassigned = cases.filter((item) => !item.assigned_to);

  const grouped = new Map<string, DashboardQueueItem[]>();

  assigned.forEach((item) => {
    const name = item.assigned_to ?? "Unassigned";
    const current = grouped.get(name) ?? [];
    current.push(item);
    grouped.set(name, current);
  });

  return (
    <div className="dashboard">
      <div className="page-header">
        <div>
          <span className="eyebrow">SYSTEM</span>
          <h1>Investigators</h1>
          <p>
            Review investigator workload and assigned risk cases.
          </p>
        </div>

        <button
          className="refresh-button"
          onClick={() => refetch()}
          disabled={isLoading}
        >
          <RefreshCw size={15} className={isLoading ? "spin" : ""} />
          {isLoading ? "Refreshing" : "Refresh"}
        </button>
      </div>

      <div className="stats-grid">
        <StatCard
          label="ASSIGNED CASES"
          value={assigned.length}
          detail="Cases currently assigned"
          icon={Users}
        />
        <StatCard
          label="UNASSIGNED"
          value={unassigned.length}
          detail="Cases awaiting assignment"
          icon={FileSearch}
          tone="warning"
        />
        <StatCard
          label="INVESTIGATORS"
          value={grouped.size}
          detail="Investigators with visible cases"
          icon={Activity}
        />
        <StatCard
          label="TOTAL QUEUE"
          value={cases.length}
          detail="Cases in the current queue"
          icon={ShieldAlert}
        />
      </div>

      {isError ? (
        <div className="empty-state page-empty">
          <XCircle size={30} />
          <strong>Investigator data unavailable</strong>
          <span>Unable to load investigator workload.</span>
          <button className="refresh-button" onClick={() => refetch()}>
            Retry
          </button>
        </div>
      ) : (
        <div className="dashboard-grid">
          {Array.from(grouped.entries()).map(([name, investigatorCases]) => (
            <div className="panel" key={name}>
              <div className="panel-heading">
                <div>
                  <span className="panel-eyebrow">INVESTIGATOR</span>
                  <h2>{name}</h2>
                </div>
                <Users size={17} />
              </div>

              <div className="case-list">
                {investigatorCases.slice(0, 8).map((item) => (
                  <button
                    className="case-row"
                    key={item.case_id}
                    onClick={() =>
                      navigate(
                        `/cases/${encodeURIComponent(item.case_id)}`,
                      )
                    }
                  >
                    <div className="case-row-main">
                      <span className="case-id">{item.case_id}</span>
                      <span className="case-reason">
                        {item.primary_reason || "Risk investigation"}
                      </span>
                    </div>

                    <div className="case-row-risk">
                      <span
                        className={`priority-badge ${priorityClass(
                          item.priority,
                        )}`}
                      >
                        {item.priority}
                      </span>
                      <strong>{formatScore(item.risk_score)}</strong>
                      <span className="case-arrow">→</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ))}

          {grouped.size === 0 && (
            <div className="panel">
              <div className="empty-state">
                <Users size={28} />
                <strong>No assigned investigators</strong>
                <span>
                  Investigator assignments will appear here as cases are
                  assigned.
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ============================================================
   ACTIVITY
============================================================ */

function ActivityPage() {
  const navigate = useNavigate();

  const {
    data,
    isLoading,
    isError,
    refetch,
  } = useQuery({
    queryKey: ["activity", "recent"],
    queryFn: () => fetchDashboardActivity(50),
  });

  const items = data?.items ?? [];

  return (
    <div className="dashboard">
      <div className="page-header">
        <div>
          <span className="eyebrow">SYSTEM</span>
          <h1>Activity</h1>
          <p>
            Review recent investigator and case activity.
          </p>
        </div>

        <button
          className="refresh-button"
          onClick={() => refetch()}
          disabled={isLoading}
        >
          <RefreshCw size={15} className={isLoading ? "spin" : ""} />
          {isLoading ? "Refreshing" : "Refresh"}
        </button>
      </div>

      <div className="panel full-panel">
        {isLoading ? (
          <div className="loading-state">
            <Loader2 size={20} className="spin" />
            Loading activity...
          </div>
        ) : isError ? (
          <div className="empty-state">
            <XCircle size={28} />
            <strong>Activity unavailable</strong>
            <span>Unable to load recent system activity.</span>
            <button className="refresh-button" onClick={() => refetch()}>
              Retry
            </button>
          </div>
        ) : items.length === 0 ? (
          <div className="empty-state">
            <Clock3 size={28} />
            <strong>No recent activity</strong>
            <span>Case actions will appear here as they occur.</span>
          </div>
        ) : (
          <div className="activity-list">
            {items.map((item) => (
              <button
                className="activity-row"
                key={`${item.case_id}-${item.timestamp}`}
                onClick={() =>
                  navigate(
                    `/cases/${encodeURIComponent(item.case_id)}`,
                  )
                }
                style={{
                  width: "100%",
                  border: 0,
                  background: "transparent",
                  textAlign: "left",
                  cursor: "pointer",
                }}
              >
                <div className="activity-icon">
                  {item.action === "CASE_RESOLVED" ? (
                    <CheckCircle2 size={15} />
                  ) : item.action === "CASE_ESCALATED" ? (
                    <AlertTriangle size={15} />
                  ) : (
                    <Activity size={15} />
                  )}
                </div>

                <div className="activity-copy">
                  <strong>{item.case_id}</strong>
                  <span>
                    {item.action
                      .replaceAll("_", " ")
                      .toLowerCase()
                      .replace(/^./, (char) => char.toUpperCase())}
                  </span>
                </div>

                <time>{formatDate(item.timestamp)}</time>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ============================================================
   SETTINGS
============================================================ */

function SettingsPage() {
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [compactMode, setCompactMode] = useState(false);
  const [saved, setSaved] = useState(false);

  const saveSettings = () => {
    localStorage.setItem(
      "razorguard.settings",
      JSON.stringify({ autoRefresh, compactMode }),
    );
    setSaved(true);
    window.setTimeout(() => setSaved(false), 2200);
  };

  return (
    <div className="dashboard">
      <div className="page-header">
        <div>
          <span className="eyebrow">SYSTEM</span>
          <h1>Settings</h1>
          <p>
            Configure investigator workspace preferences.
          </p>
        </div>

        <button className="refresh-button" onClick={saveSettings}>
          <CheckCircle2 size={15} />
          {saved ? "Saved" : "Save Settings"}
        </button>
      </div>

      <div className="dashboard-grid">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <span className="panel-eyebrow">WORKSPACE</span>
              <h2>Interface preferences</h2>
            </div>
            <Settings size={17} />
          </div>

          <div className="signal-grid">
            <button
              className={`network-signal ${autoRefresh ? "active" : ""}`}
              onClick={() => setAutoRefresh((value) => !value)}
              style={{ border: 0, textAlign: "left", cursor: "pointer" }}
            >
              {autoRefresh ? (
                <CheckCircle2 size={18} />
              ) : (
                <XCircle size={18} />
              )}
              <div>
                <strong>Automatic refresh</strong>
                <span>
                  {autoRefresh
                    ? "Enabled for operational data"
                    : "Disabled"}
                </span>
              </div>
            </button>

            <button
              className={`network-signal ${compactMode ? "active" : ""}`}
              onClick={() => setCompactMode((value) => !value)}
              style={{ border: 0, textAlign: "left", cursor: "pointer" }}
            >
              {compactMode ? (
                <CheckCircle2 size={18} />
              ) : (
                <XCircle size={18} />
              )}
              <div>
                <strong>Compact workspace</strong>
                <span>
                  {compactMode
                    ? "Enabled"
                    : "Standard density"}
                </span>
              </div>
            </button>
          </div>
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <span className="panel-eyebrow">SESSION</span>
              <h2>Analyst profile</h2>
            </div>
            <Users size={17} />
          </div>

          <div className="signal-grid">
            <div>
              <span>Role</span>
              <strong>Security Analyst</strong>
            </div>
            <div>
              <span>Workspace</span>
              <strong>Fraud Operations</strong>
            </div>
            <div>
              <span>Engine</span>
              <strong>Operational</strong>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   APP SHELL
============================================================ */

function AppShell() {
  const [helpOpen, setHelpOpen] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const navigate = useNavigate();

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <ShieldAlert
              size={20}
              strokeWidth={2.4}
            />
          </div>

          <div>
            <div className="brand-name">
              RazorGuard
            </div>

            <div className="brand-subtitle">
              Risk Intelligence
            </div>
          </div>
        </div>

        <div className="workspace">
          <div className="workspace-label">
            WORKSPACE
          </div>

          <button className="workspace-selector">
            <span className="workspace-dot" />

            <span>
              Fraud Operations
            </span>

            <ChevronDown size={15} />
          </button>
        </div>

        <nav className="navigation">
          <div className="nav-section-label">
            OPERATIONS
          </div>

          {navigation.map(
            (item) => {
              const Icon = item.icon;

              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  end={
                    item.path === "/"
                  }
                  className={({
                    isActive,
                  }) =>
                    `nav-item ${
                      isActive
                        ? "active"
                        : ""
                    }`
                  }
                >
                  <Icon size={18} />

                  <span>
                    {item.label}
                  </span>
                </NavLink>
              );
            },
          )}

          <div className="nav-section-label secondary">
            SYSTEM
          </div>

          {secondaryNavigation.map(
            (item) => {
              const Icon = item.icon;

              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) =>
                    `nav-item ${isActive ? "active" : ""}`
                  }
                >
                  <Icon size={18} />

                  <span>
                    {item.label}
                  </span>
                </NavLink>
              );
            },
          )}
        </nav>

        <div className="sidebar-footer">
          <div className="system-status">
            <span className="status-dot" />

            <span>
              Risk engine operational
            </span>
          </div>

          <div className="version">
            RazorGuard v0.2.0
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="topbar-left">
            <div className="breadcrumb">
              FRAUD OPERATIONS
            </div>
          </div>

          <div
            className="topbar-actions"
            style={{
              position: "relative",
              display: "flex",
              alignItems: "center",
            }}
          >
            <button
              className="icon-button"
              aria-label="Help"
              onClick={() => {
                setHelpOpen((value) => !value);
                setNotificationsOpen(false);
                setProfileOpen(false);
              }}
            >
              <CircleHelp size={18} />
            </button>

            <button
              className="icon-button notification-button"
              aria-label="Notifications"
              onClick={() => {
                setNotificationsOpen((value) => !value);
                setHelpOpen(false);
                setProfileOpen(false);
              }}
            >
              <Bell size={18} />
              <span className="notification-dot" />
            </button>

            <button
              className="user-menu"
              aria-label="Analyst menu"
              onClick={() => {
                setProfileOpen((value) => !value);
                setHelpOpen(false);
                setNotificationsOpen(false);
              }}
              style={{
                border: 0,
                background: "transparent",
                cursor: "pointer",
              }}
            >
              <div className="avatar">SA</div>

              <div className="user-details">
                <span className="user-name">Security Analyst</span>
                <span className="user-role">Investigator</span>
              </div>

              <ChevronDown size={15} />
            </button>

            {helpOpen && (
              <div
                className="topbar-popover"
                style={{
                  position: "absolute",
                  top: "calc(100% + 12px)",
                  right: 0,
                  width: 340,
                  maxWidth: "calc(100vw - 32px)",
                  padding: "16px",
                  display: "flex",
                  flexDirection: "column",
                  gap: "8px",
                  zIndex: 1000,
                  boxSizing: "border-box",
                  border: "1px solid var(--border, #243140)",
                  borderRadius: "10px",
                  background: "var(--panel, #0d141c)",
                  boxShadow: "0 18px 40px rgba(0, 0, 0, 0.35)",
                }}
              >
                <strong style={{ display: "block" }}>RazorGuard Help</strong>
                <span>
                  Use Case Queue to investigate cases, Network Intelligence
                  to trace relationships, and Risk Analytics to inspect
                  engine behavior.
                </span>
              </div>
            )}

            {notificationsOpen && (
              <div
                className="topbar-popover"
                style={{
                  position: "absolute",
                  top: "calc(100% + 12px)",
                  right: 0,
                  width: 340,
                  maxWidth: "calc(100vw - 32px)",
                  padding: "16px",
                  display: "flex",
                  flexDirection: "column",
                  gap: "8px",
                  zIndex: 1000,
                  boxSizing: "border-box",
                  border: "1px solid var(--border, #243140)",
                  borderRadius: "10px",
                  background: "var(--panel, #0d141c)",
                  boxShadow: "0 18px 40px rgba(0, 0, 0, 0.35)",
                }}
              >
                <strong style={{ display: "block" }}>Notifications</strong>
                <span>
                  Review recent case activity from the Activity workspace.
                </span>
                <button
                  className="text-button"
                  onClick={() => {
                    setNotificationsOpen(false);
                    navigate("/activity");
                  }}
                >
                  Open Activity
                </button>
              </div>
            )}

            {profileOpen && (
              <div
                className="topbar-popover"
                style={{
                  position: "absolute",
                  top: "calc(100% + 12px)",
                  right: 0,
                  width: 340,
                  maxWidth: "calc(100vw - 32px)",
                  padding: "16px",
                  display: "flex",
                  flexDirection: "column",
                  gap: "8px",
                  zIndex: 1000,
                  boxSizing: "border-box",
                  border: "1px solid var(--border, #243140)",
                  borderRadius: "10px",
                  background: "var(--panel, #0d141c)",
                  boxShadow: "0 18px 40px rgba(0, 0, 0, 0.35)",
                }}
              >
                <strong style={{ display: "block" }}>Security Analyst</strong>
                <span>Investigator · Fraud Operations</span>
                <button
                  className="text-button"
                  onClick={() => {
                    setProfileOpen(false);
                    navigate("/settings");
                  }}
                >
                  Open Settings
                </button>
              </div>
            )}
          </div>
        </header>

        <section className="content">
          <Routes>
            <Route
              path="/"
              element={<Dashboard />}
            />

            <Route
              path="/cases"
              element={<Cases />}
            />

            <Route
              path="/cases/:caseId"
              element={<InvestigationRoute />}
            />

            <Route
              path="/network"
              element={<NetworkIntelligence />}
            />

            <Route
              path="/analytics"
              element={<RiskAnalytics />}
            />

            <Route
              path="/investigators"
              element={<Investigators />}
            />

            <Route
              path="/activity"
              element={<ActivityPage />}
            />

            <Route
              path="/settings"
              element={<SettingsPage />}
            />

            <Route
              path="*"
              element={
                <Navigate
                  to="/"
                  replace
                />
              }
            />
          </Routes>
        </section>
      </main>
    </div>
  );
}

/* ============================================================
   INVESTIGATION ROUTE
============================================================ */

function InvestigationRoute() {
  const caseId =
    window.location.pathname.split(
      "/cases/",
    )[1];

  return (
    <Investigation
      caseId={
        caseId
          ? decodeURIComponent(caseId)
          : undefined
      }
    />
  );
}

/* ============================================================
   APP
============================================================ */

export default function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  );
}