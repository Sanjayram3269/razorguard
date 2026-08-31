import axios from "axios";

/* ============================================================
   API CLIENT
============================================================ */

export const api = axios.create({
  baseURL: "/api",
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 15_000,
});


/* ============================================================
   CASE TYPES
============================================================ */

export type CaseStatus =
  | "OPEN"
  | "IN_REVIEW"
  | "ESCALATED"
  | "RESOLVED"
  | "DISMISSED";

export type CasePriority =
  | "CRITICAL"
  | "HIGH"
  | "MEDIUM"
  | "LOW";


export interface CaseRecord {
  case_id: string;
  transaction_id: string;

  status: CaseStatus | string;
  priority: CasePriority | string;
  assigned_to: string | null;

  created_at: string;
  updated_at: string;

  risk_score: number;
  risk_level: string;
  decision: string;

  primary_reason: string;
  evidence_text: string;

  model_probability: number;
  network_score: number;

  investigation_narrative: string;
}


export interface CaseListResponse {
  cases: CaseRecord[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}


/* ============================================================
   DASHBOARD SUMMARY
============================================================ */

export interface DashboardSummary {
  open_cases: number;
  critical_cases: number;
  high_cases: number;
  medium_cases: number;
  low_cases: number;
  average_risk_score: number;
  total_cases: number;
}
/* ============================================================
   RISK ANALYTICS
============================================================ */

export interface AnalyticsDistributionItem {
  label: string;
  count: number;
  percentage: number;
}

export interface AnalyticsMetricResponse {
  total_cases: number;
  average_risk_score: number;
  median_risk_score: number;
  maximum_risk_score: number;
  average_model_probability: number;
  average_network_score: number;

  priority_distribution: AnalyticsDistributionItem[];
  risk_level_distribution: AnalyticsDistributionItem[];
  decision_distribution: AnalyticsDistributionItem[];
  status_distribution: AnalyticsDistributionItem[];
  top_reasons: AnalyticsDistributionItem[];
}

export async function fetchAnalyticsOverview(): Promise<AnalyticsMetricResponse> {
  const response =
    await api.get<AnalyticsMetricResponse>(
      "/v1/analytics/overview",
    );

  return response.data;
}

export interface NetworkSummary {
  accounts: number;
  devices: number;
  merchants: number;

  account_device_edges: number;
  account_merchant_edges: number;
  device_merchant_edges: number;
}

export interface NetworkRiskSignals {
  device_shared: boolean;
  merchant_shared: boolean;
  new_device_for_account: boolean;
  new_merchant_for_account: boolean;
}

export interface NetworkTransaction {
  transaction_id: string;
  timestamp: string;

  account_id: string;
  device_id: string;
  merchant_id: string;

  account_history_count: number;

  accounts_seen_on_device: string[];
  accounts_seen_at_merchant: string[];

  related_transaction_count: number;

  network_risk_signals: NetworkRiskSignals;
}

export interface RiskClusterSignal {
  type: string;
  severity: string;
  value: number;
  evidence: string;
}

export interface RiskClusterTimelineItem {
  transaction_id: string;
  timestamp: string;
  account_id: string;
  device_id: string;
  merchant_id: string;
}

export interface RiskClusterResponse {
  cluster_id: string;
  cluster_type: string;
  risk_score: number;

  accounts: string[];
  devices: string[];
  merchants: string[];
  transactions: string[];

  signals: RiskClusterSignal[];
  evidence: string[];
  timeline: RiskClusterTimelineItem[];
}

export async function fetchNetworkSummary(): Promise<NetworkSummary> {
  const response = await api.get<NetworkSummary>(
    "/v1/network/summary",
  );

  return response.data;
}

export async function fetchNetworkTransaction(
  transactionId: string,
): Promise<NetworkTransaction> {
  const response =
    await api.get<NetworkTransaction>(
      `/v1/network/transaction/${encodeURIComponent(
        transactionId,
      )}`,
    );

  return response.data;
}

/* ============================================================
   COORDINATED RISK CLUSTER
============================================================ */

export async function fetchNetworkCluster(
  transactionId: string,
): Promise<RiskClusterResponse> {
  const response =
    await api.get<RiskClusterResponse>(
      `/v1/network/transaction/${encodeURIComponent(
        transactionId,
      )}/cluster`,
    );

  return response.data;
}

/* ============================================================
   DASHBOARD DISTRIBUTION
============================================================ */

export interface DashboardDistributionItem {
  label: string;
  count: number;
  percentage: number;
}


export interface DashboardDistributionResponse {
  items: DashboardDistributionItem[];
  total: number;
}


/* ============================================================
   DASHBOARD ACTIVITY
============================================================ */

export interface DashboardActivityItem {
  case_id: string;
  transaction_id: string;
  action: string;
  actor: string;
  timestamp: string;
  details: string;
}


export interface DashboardActivityResponse {
  items: DashboardActivityItem[];
  total: number;
}


/* ============================================================
   DASHBOARD QUEUE
============================================================ */

export interface DashboardQueueItem {
  case_id: string;
  transaction_id: string;
  priority: string;
  risk_score: number;
  risk_level: string;
  decision: string;
  primary_reason: string;
}


export interface DashboardQueueResponse {
  items: DashboardQueueItem[];
  total: number;
}


/* ============================================================
   CASE FILTERS
============================================================ */

export interface CaseFilters {
  status?: CaseStatus | string;
  assigned_to?: string;
  priority?: CasePriority | string;

  search?: string;

  sort_by?: string;
  sort_order?: "asc" | "desc";

  page?: number;
  page_size?: number;
}


/* ============================================================
   CASE MANAGEMENT REQUESTS
============================================================ */

export interface AssignCaseRequest {
  investigator: string;
  actor?: string;
}


export interface TransitionCaseRequest {
  status: CaseStatus | string;
  actor?: string;
  details?: string;
}


/* ============================================================
   AUDIT
============================================================ */

export interface AuditEvent {
  case_id: string;
  timestamp: string;
  action: string;
  actor: string;

  from_status: string | null;
  to_status: string | null;

  details: string;
}


export interface AuditResponse {
  case_id: string;
  events: AuditEvent[];
  total: number;
}


/* ============================================================
   TRANSACTION SCORING
============================================================ */

export interface TransactionScoreRequest {
  transaction_id: string;
  account_id: string;
  merchant_id: string;
  device_id: string;

  timestamp: string;

  amount: number;

  ip_country: string;
  shipping_country: string;
  payment_method: string;
  merchant_category: string;
}


export interface TransactionScoreResponse {
  transaction_id: string;

  risk_score: number;
  risk_level: string;
  decision: string;

  primary_reason: string;
  evidence: string[];

  model_probability: number;
  network_score: number;
  behavioral_signal: number;

  model: string;
  model_threshold: number;

  case_id?: string | null;
}


/* ============================================================
   API ERRORS
============================================================ */

export interface ApiError {
  detail: string;
}


/* ============================================================
   ERROR NORMALIZATION
============================================================ */

export function getApiErrorMessage(
  error: unknown,
  fallback = "Something went wrong.",
): string {
  if (axios.isAxiosError<ApiError>(error)) {
    const detail = error.response?.data?.detail;

    if (typeof detail === "string" && detail.trim()) {
      return detail;
    }

    if (error.response?.status === 404) {
      return "The requested resource was not found.";
    }

    if (error.response?.status === 409) {
      return "The requested operation conflicts with the current case state.";
    }

    if (error.response?.status === 422) {
      return "The submitted data is invalid.";
    }

    if (error.response?.status === 503) {
      return "RazorGuard is temporarily unavailable.";
    }

    if (error.code === "ECONNABORTED") {
      return "The RazorGuard API request timed out.";
    }

    if (!error.response) {
      return "Unable to reach the RazorGuard API.";
    }
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  return fallback;
}


/* ============================================================
   CASES
============================================================ */

export async function fetchCases(): Promise<CaseListResponse> {
  const response = await api.get<CaseListResponse>(
    "/v1/cases",
  );

  return response.data;
}


export async function fetchCasesFiltered(
  filters: CaseFilters = {},
): Promise<CaseListResponse> {
  const params: Record<string, string | number> = {};

  if (filters.status) {
    params.status = filters.status;
  }

  if (filters.assigned_to) {
    params.assigned_to = filters.assigned_to;
  }

  if (filters.priority) {
    params.priority = filters.priority;
  }

  if (filters.search?.trim()) {
    params.search = filters.search.trim();
  }

  if (filters.sort_by) {
    params.sort_by = filters.sort_by;
  }

  if (filters.sort_order) {
    params.sort_order = filters.sort_order;
  }

  if (filters.page) {
    params.page = filters.page;
  }

  if (filters.page_size) {
    params.page_size = filters.page_size;
  }

  const response = await api.get<CaseListResponse>(
    "/v1/cases",
    {
      params,
    },
  );

  return response.data;
}


export async function fetchCase(
  caseId: string,
): Promise<CaseRecord> {
  const response = await api.get<CaseRecord>(
    `/v1/cases/${encodeURIComponent(caseId)}`,
  );

  return response.data;
}


/* ============================================================
   DASHBOARD SUMMARY
============================================================ */

export async function fetchDashboardSummary(): Promise<DashboardSummary> {
  const response = await api.get<DashboardSummary>(
    "/v1/dashboard/summary",
  );

  return response.data;
}


/* ============================================================
   DASHBOARD DISTRIBUTION
============================================================ */

export async function fetchDashboardDistribution(): Promise<DashboardDistributionResponse> {
  const response =
    await api.get<DashboardDistributionResponse>(
      "/v1/dashboard/distribution",
    );

  return response.data;
}


/* ============================================================
   DASHBOARD ACTIVITY
============================================================ */

export async function fetchDashboardActivity(
  limit = 10,
): Promise<DashboardActivityResponse> {
  const response =
    await api.get<DashboardActivityResponse>(
      "/v1/dashboard/activity",
      {
        params: {
          limit,
        },
      },
    );

  return response.data;
}


/* ============================================================
   DASHBOARD QUEUE
============================================================ */

export async function fetchDashboardQueue(
  limit = 10,
): Promise<DashboardQueueResponse> {
  const response =
    await api.get<DashboardQueueResponse>(
      "/v1/dashboard/queue",
      {
        params: {
          limit,
        },
      },
    );

  return response.data;
}


/* ============================================================
   CASE ASSIGNMENT
============================================================ */

export async function assignCase(
  caseId: string,
  request: AssignCaseRequest,
): Promise<CaseRecord> {
  const response = await api.post<CaseRecord>(
    `/v1/cases/${encodeURIComponent(caseId)}/assign`,
    request,
  );

  return response.data;
}


/* ============================================================
   CASE STATUS TRANSITION
============================================================ */

export async function transitionCase(
  caseId: string,
  request: TransitionCaseRequest,
): Promise<CaseRecord> {
  const response = await api.post<CaseRecord>(
    `/v1/cases/${encodeURIComponent(caseId)}/transition`,
    request,
  );

  return response.data;
}


/* ============================================================
   AUDIT HISTORY
============================================================ */

export async function fetchCaseAudit(
  caseId: string,
): Promise<AuditResponse> {
  const response =
    await api.get<AuditResponse>(
      `/v1/cases/${encodeURIComponent(caseId)}/audit`,
    );

  return response.data;
}


/* ============================================================
   TRANSACTION SCORING
============================================================ */

export async function scoreTransaction(
  request: TransactionScoreRequest,
): Promise<TransactionScoreResponse> {
  const response =
    await api.post<TransactionScoreResponse>(
      "/v1/transactions/score",
      request,
    );

  return response.data;
}