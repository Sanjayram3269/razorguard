import {
  ArrowRight,
  Building2,
  CreditCard,
  Network,
  Smartphone,
  Store,
} from "lucide-react";

import type { NetworkTransaction } from "../lib/api";

interface EvidenceGraphProps {
  transaction: NetworkTransaction;
}

export default function EvidenceGraph({
  transaction,
}: EvidenceGraphProps) {
  const signals =
    transaction.network_risk_signals;

  const merchantAccounts =
    transaction.accounts_seen_at_merchant.length;

  const deviceAccounts =
    transaction.accounts_seen_on_device.length;

  return (
    <div className="evidence-graph">
      <div className="evidence-graph-header">
        <div>
          <span className="panel-eyebrow">
            EVIDENCE GRAPH
          </span>

          <h2>
            Transaction relationship map
          </h2>

          <p>
            Connected entities and network signals
            surrounding this transaction.
          </p>
        </div>

        <Network size={19} />
      </div>

      <div className="evidence-graph-canvas">
        <div className="evidence-node transaction-node">
          <CreditCard size={19} />

          <span>
            TRANSACTION
          </span>

          <strong>
            {transaction.transaction_id}
          </strong>
        </div>

        <div className="graph-connector connector-left">
          <ArrowRight size={15} />
        </div>

        <div className="evidence-node">
          <Building2 size={18} />

          <span>
            ACCOUNT
          </span>

          <strong>
            {transaction.account_id}
          </strong>

          <small>
            {transaction.account_history_count} historical
            transactions
          </small>
        </div>

        <div className="graph-connector connector-device">
          <ArrowRight size={15} />
        </div>

        <div
          className={`evidence-node ${
            signals.new_device_for_account
              ? "risk-node"
              : ""
          }`}
        >
          <Smartphone size={18} />

          <span>
            DEVICE
          </span>

          <strong>
            {transaction.device_id}
          </strong>

          {signals.new_device_for_account && (
            <small>
              New device for account
            </small>
          )}
        </div>

        <div className="graph-connector connector-merchant">
          <ArrowRight size={15} />
        </div>

        <div
          className={`evidence-node ${
            signals.merchant_shared
              ? "risk-node"
              : ""
          }`}
        >
          <Store size={18} />

          <span>
            MERCHANT
          </span>

          <strong>
            {transaction.merchant_id}
          </strong>

          {signals.merchant_shared && (
            <small>
              Shared across{" "}
              {merchantAccounts.toLocaleString()} accounts
            </small>
          )}
        </div>
      </div>

      <div className="graph-evidence-grid">
        <div className="graph-evidence-card">
          <span>
            MERCHANT CONNECTION
          </span>

          <strong>
            {merchantAccounts.toLocaleString()}
          </strong>

          <p>
            accounts observed at this merchant
          </p>
        </div>

        <div className="graph-evidence-card">
          <span>
            RELATED TRANSACTIONS
          </span>

          <strong>
            {transaction.related_transaction_count.toLocaleString()}
          </strong>

          <p>
            transactions connected to this activity
          </p>
        </div>

        <div className="graph-evidence-card">
          <span>
            DEVICE CONNECTION
          </span>

          <strong>
            {deviceAccounts.toLocaleString()}
          </strong>

          <p>
            additional accounts sharing this device
          </p>
        </div>
      </div>

      <div className="graph-signal-strip">
        <span className="panel-eyebrow">
          NETWORK SIGNALS
        </span>

        <div className="graph-signal-list">
          <div
            className={
              signals.merchant_shared
                ? "graph-signal active"
                : "graph-signal"
            }
          >
            <span>
              Merchant shared
            </span>

            <strong>
              {signals.merchant_shared
                ? "YES"
                : "NO"}
            </strong>
          </div>

          <div
            className={
              signals.device_shared
                ? "graph-signal active"
                : "graph-signal"
            }
          >
            <span>
              Device shared
            </span>

            <strong>
              {signals.device_shared
                ? "YES"
                : "NO"}
            </strong>
          </div>

          <div
            className={
              signals.new_device_for_account
                ? "graph-signal active"
                : "graph-signal"
            }
          >
            <span>
              New device
            </span>

            <strong>
              {signals.new_device_for_account
                ? "YES"
                : "NO"}
            </strong>
          </div>

          <div
            className={
              signals.new_merchant_for_account
                ? "graph-signal active"
                : "graph-signal"
            }
          >
            <span>
              New merchant
            </span>

            <strong>
              {signals.new_merchant_for_account
                ? "YES"
                : "NO"}
            </strong>
          </div>
        </div>
      </div>
    </div>
  );
}