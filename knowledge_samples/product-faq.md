# Apex Engine: Cloud Orchestrator Product Specifications and FAQ

## 1. Platform Overview and Subscription Tiers
The Apex Engine platform provides distributed job execution and telemetry streaming for microservice clusters. Subscription pricing and quotas are structured across three service tiers:

| Plan Tier | Monthly Cost | Max Projects | Active Storage | API Request Rate | Guaranteed SLA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Starter Tier | $49 | 5 projects | 50 GB | 100 req/min | 99.5% |
| Professional Tier | $199 | 25 projects | 250 GB | 500 req/min | 99.9% |
| Enterprise Tier | $899 | Unlimited | 2 TB | 2500 req/min | 99.99% |

Plan upgrades take effect immediately with pro-rated billing. Downgrades take effect at the conclusion of the active 30-day billing cycle.

## 2. Webhook Ingestion and Retry Policy
Outbound webhook notifications are dispatched upon task state transitions (e.g., `QUEUED`, `RUNNING`, `TERMINATED`):
- Delivery failures trigger a maximum of 5 retries with exponential backoff over a 24-hour delivery window.
- Payload signatures are computed using HMAC-SHA256 with user-configured endpoint secrets.
- Webhook endpoints failing consecutively for 72 hours are automatically disabled.

## 3. Supported Deployment Regions and Data Retention
Apex Engine instances are available across three regional zones:
- `us-east-2` (Ohio Primary Hub)
- `eu-west-1` (Dublin Resiliency Node)
- `ap-southeast-1` (Singapore Cluster)

Audit logs are retained for 90 days on Professional tier accounts and 7 years on Enterprise tier accounts.

## 4. Bulk Asset Export Architecture
Data exports can be exported into JSON, CSV, or Apache Parquet formats directly from the administrative portal. Bulk asset exports exceeding 10,000 entities trigger an asynchronous job token with a cryptographic TTL of 43,200 seconds before being purged from temporary export caches. Export generation does not consume customer runtime API quotas.
