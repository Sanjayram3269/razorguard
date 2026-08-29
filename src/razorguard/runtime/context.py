from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class RuntimeContextStore:
    """
    Read-only transaction history provider for runtime scoring.

    The store exposes only transactions that occurred strictly before
    the requested timestamp. It never mutates the underlying history.

    This keeps runtime inference aligned with RazorGuard's
    point-in-time feature engineering guarantees.
    """

    REQUIRED_COLUMNS = {
        "transaction_id",
        "account_id",
        "merchant_id",
        "device_id",
        "timestamp",
        "amount",
        "ip_country",
        "shipping_country",
        "payment_method",
        "merchant_category",
    }

    def __init__(
        self,
        transactions_path: str | Path,
    ) -> None:
        self.transactions_path = Path(
            transactions_path
        )

        if not self.transactions_path.exists():
            raise FileNotFoundError(
                "Transaction history not found: "
                f"{self.transactions_path}"
            )

        self._transactions = pd.read_parquet(
            self.transactions_path
        )

        missing = (
            self.REQUIRED_COLUMNS
            - set(self._transactions.columns)
        )

        if missing:
            raise ValueError(
                "Transaction history is missing required "
                f"columns: {sorted(missing)}"
            )

        self._transactions = (
            self._transactions
            .copy()
        )

        self._transactions["timestamp"] = (
            pd.to_datetime(
                self._transactions["timestamp"]
            )
        )

        self._transactions = (
            self._transactions
            .sort_values(
                "timestamp",
                kind="stable",
            )
            .reset_index(drop=True)
        )

    @property
    def transactions(self) -> pd.DataFrame:
        """
        Return a defensive copy of the transaction history.
        """

        return self._transactions.copy()

    def get_transaction_history(
        self,
        before_timestamp: Any,
    ) -> pd.DataFrame:
        """
        Return all transactions strictly before a timestamp.
        """

        timestamp = pd.Timestamp(
            before_timestamp
        )

        return (
            self._transactions[
                self._transactions["timestamp"]
                < timestamp
            ]
            .copy()
            .reset_index(drop=True)
        )

    def get_account_history(
        self,
        account_id: str,
        before_timestamp: Any,
    ) -> pd.DataFrame:
        """
        Return prior transactions for one account.
        """

        history = self.get_transaction_history(
            before_timestamp
        )

        return (
            history[
                history["account_id"]
                == account_id
            ]
            .copy()
            .reset_index(drop=True)
        )

    def get_device_history(
        self,
        device_id: str,
        before_timestamp: Any,
    ) -> pd.DataFrame:
        """
        Return prior transactions for one device.
        """

        history = self.get_transaction_history(
            before_timestamp
        )

        return (
            history[
                history["device_id"]
                == device_id
            ]
            .copy()
            .reset_index(drop=True)
        )

    def get_merchant_history(
        self,
        merchant_id: str,
        before_timestamp: Any,
    ) -> pd.DataFrame:
        """
        Return prior transactions for one merchant.
        """

        history = self.get_transaction_history(
            before_timestamp
        )

        return (
            history[
                history["merchant_id"]
                == merchant_id
            ]
            .copy()
            .reset_index(drop=True)
        )

    def build_scoring_frame(
        self,
        transaction: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Build a scoring dataframe containing historical context
        followed by the current transaction.

        The current transaction is appended LAST.

        This is critical because RazorGuard's feature engine
        calculates point-in-time features sequentially and only
        incorporates the current transaction after calculating
        its historical features.
        """

        required = {
            "transaction_id",
            "account_id",
            "merchant_id",
            "device_id",
            "timestamp",
            "amount",
            "ip_country",
            "shipping_country",
            "payment_method",
            "merchant_category",
        }

        missing = required - set(
            transaction.keys()
        )

        if missing:
            raise ValueError(
                "Transaction is missing required fields: "
                f"{sorted(missing)}"
            )

        timestamp = pd.Timestamp(
            transaction["timestamp"]
        )

        history = self.get_transaction_history(
            timestamp
        )

        current = pd.DataFrame(
            [transaction]
        )

        current["timestamp"] = pd.to_datetime(
            current["timestamp"]
        )

        # Runtime requests do not contain a chargeback label.
        if "is_chargeback" not in current.columns:
            current["is_chargeback"] = 0

        history_for_features = history.copy()

        if "is_chargeback" not in history_for_features.columns:
            history_for_features[
                "is_chargeback"
            ] = 0

        frame = pd.concat(
            [
                history_for_features,
                current,
            ],
            ignore_index=True,
        )

        return (
            frame
            .sort_values(
                "timestamp",
                kind="stable",
            )
            .reset_index(drop=True)
        )

    def get_context_summary(
        self,
        transaction: dict[str, Any],
    ) -> dict[str, int]:
        """
        Return compact historical context metrics for a transaction.
        """

        timestamp = pd.Timestamp(
            transaction["timestamp"]
        )

        account_history = (
            self.get_account_history(
                transaction["account_id"],
                timestamp,
            )
        )

        device_history = (
            self.get_device_history(
                transaction["device_id"],
                timestamp,
            )
        )

        merchant_history = (
            self.get_merchant_history(
                transaction["merchant_id"],
                timestamp,
            )
        )

        return {
            "prior_transactions": int(
                len(
                    self.get_transaction_history(
                        timestamp
                    )
                )
            ),
            "prior_account_transactions": int(
                len(account_history)
            ),
            "prior_device_transactions": int(
                len(device_history)
            ),
            "prior_merchant_transactions": int(
                len(merchant_history)
            ),
        }