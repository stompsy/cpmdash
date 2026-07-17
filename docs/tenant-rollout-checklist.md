# Tenant Isolation Rollout Checklist

This checklist covers rollout and verification for county/agency tenant isolation.

## Pre-Deploy

1. Confirm branch is green:
- make lint
- make type
- make test

2. Confirm migrations include tenant schema and seed data:
- core migrations: County/Agency and core model agency FKs
- accounts migration: user agency FK
- data_import migration: batch and staging agency/county fields

3. Verify seeded defaults exist in a fresh migrated database:
- County: clallam-county
- Agencies: port-angeles, sequim

4. Snapshot production database before migration.

5. Run migration dry run against staging copy of production data.

## Deploy Step 1: Schema + Data Migrations

1. Deploy migration artifacts.
2. Run migrations.
3. Verify migration completion with no failed operations.

## Deploy Step 2: App Code

1. Deploy app code with tenant scope enforcement.
2. Restart app processes.

## Post-Deploy Verification

1. Authentication and scope:
- Log in as agency-scoped staff user.
- Confirm dashboard pages load and show only agency data by default.
- Confirm county mode aggregates only same-county agencies.

2. Dashboard security checks:
- Attempt query-param tampering with another agency_id.
- Expected: access is constrained to authorized tenant scope.

3. Data import checks:
- Upload requires county + agency selection.
- Create batch with valid tenant, process, review, and commit successfully.
- Confirm committed rows are stamped with batch agency.

4. Admin checks:
- Non-superuser only sees rows for their agency in core and import admin pages.
- Superuser can see all rows.

5. Logging checks:
- Non-superuser only sees own-tenant import logs/batches.

## Rollback Notes

1. If code deploy fails but migrations succeeded:
- Roll back application release only.
- Keep schema changes; deploy prior compatible app version if available.

2. If migration step fails mid-flight:
- Stop deploy.
- Restore DB snapshot if needed.
- Fix migration issue in staging before retrying production.

## Smoke Test Commands

- make lint
- make type
- make test
- uv run pytest src/apps/dashboard/tests/test_tenant_scope.py -q
- uv run pytest src/apps/data_import/tests/test_tenant_authorization.py -q
