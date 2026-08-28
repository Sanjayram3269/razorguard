# RazorGuard Data Model

The synthetic world uses four core entities.

## accounts

- `account_id`
- `created_at`
- `home_country`
- `device_id`
- `account_segment`

## transactions

- `transaction_id`
- `account_id`
- `merchant_id`
- `timestamp`
- `amount`
- `currency`
- `payment_method`
- `ip_country`
- `shipping_country`
- `device_id`
- `is_chargeback`

## chargebacks

- `transaction_id`
- `chargeback_at`
- `reason_code`

## Design rule

Runtime features may only use information available **at transaction decision time**.

Future chargeback outcomes are labels, never features.

Historical aggregates are calculated using prior events only.
