# Engineering lens — frontend, backend, systems

Use for features, bugs, APIs, UIs, data pipelines, deployments, performance,
security, and technical decisions.

## Frame

- Who is the user, and what is their path through the system?
- What is the observable success? What is the failure, and what is its blast
  radius?
- What are the constraints: latency, throughput, cost, compatibility, deadline,
  team size, existing architecture?
- Is this reversible? What is the rollback?

## Correctness

- What is the source of truth? Where can state drift?
- What are the contracts: API schema, types, invariants, error codes?
- What happens on empty, slow, partial, duplicated, reordered, or malicious
  input?
- What is idempotent? What is transactional? What happens on retry?
- What is the failure mode under concurrency, restart, and network partition?

## Frontend

- The user path: loading, success, empty, error, offline, permission-denied.
- Accessibility: keyboard, focus, labels, contrast, screen reader, reduced
  motion, touch targets.
- Performance: initial bundle, route split, images, fonts, layout shift, long
  lists, memoization, network waterfalls.
- State: local vs server vs URL; optimistic updates; cache invalidation;
  stale-while-revalidate.
- Responsive: smallest and largest viewport; overflow; zoom; localization.
- Design quality: hierarchy, spacing, typography, motion, consistency; does it
  look intentional rather than generic?

## Backend

- API design: resources, verbs, pagination, filtering, versioning, errors.
- Data: schema, indexes, migrations, constraints, backups, retention.
- Security: authn/authz, tenancy, input validation, secrets, SSRF, injection,
  rate limits, audit logs.
- Reliability: timeouts, retries with jitter, circuit breakers, queues,
  dead-letter, idempotency keys.
- Observability: logs, metrics, traces, alerts, SLOs; can you debug a
  production failure without guessing?
- Cost and scaling: hot paths, N+1, cache, batch, backpressure.

## Verification

- Write the smallest test that fails before the fix and passes after.
- Prefer a regression test over a promise.
- Test the contract, the edge, and the rollback — not only the happy path.
- Measure performance before and after; do not trust “should be faster.”
- Canary or feature-flag the risky change; know how to turn it off.

## Output

Answer first. Then the change, the test evidence, the risk, and the rollback.
Do not ship a plan without a verification step.
