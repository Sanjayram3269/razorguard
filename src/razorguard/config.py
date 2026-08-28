from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATED_DATA = ROOT / "data" / "generated"
ARTIFACTS = ROOT / "artifacts"

TRANSACTIONS_PATH = GENERATED_DATA / "transactions.parquet"
ACCOUNTS_PATH = GENERATED_DATA / "accounts.parquet"
CHARGEBACKS_PATH = GENERATED_DATA / "chargebacks.parquet"

RANDOM_SEED = 42
