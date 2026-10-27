#!/usr/bin/env bash
# Runs on every container start. Starts Postgres as usual and, once the real
# server is up, makes sure the app roles and database exist (idempotent):
#   - owner:  owns the database; used to create tables and load data
#   - reader: read-only; can SELECT from every table, including future ones
set -euo pipefail

: "${APP_DB_NAME:?APP_DB_NAME is not set}"
: "${APP_DB_OWNER:?APP_DB_OWNER is not set}"
: "${APP_DB_OWNER_PASSWORD:?APP_DB_OWNER_PASSWORD is not set}"
: "${APP_DB_READER:?APP_DB_READER is not set}"
: "${APP_DB_READER_PASSWORD:?APP_DB_READER_PASSWORD is not set}"

ensure_app_db() {
  # Wait on TCP, not the socket: the temporary server used during first-time
  # init only listens on the socket, so this waits for the real server.
  until pg_isready -q -h 127.0.0.1 -U "$POSTGRES_USER"; do sleep 1; done

  PGPASSWORD="$POSTGRES_PASSWORD" psql -q -v ON_ERROR_STOP=1 \
    -h 127.0.0.1 -U "$POSTGRES_USER" -d postgres \
    -v app_db="$APP_DB_NAME" \
    -v owner="$APP_DB_OWNER" \
    -v owner_password="$APP_DB_OWNER_PASSWORD" \
    -v reader="$APP_DB_READER" \
    -v reader_password="$APP_DB_READER_PASSWORD" <<'SQL'
-- Roles. Passwords are re-applied every start to stay in sync with .env.
SELECT format('CREATE ROLE %I', :'owner')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'owner') \gexec
SELECT format('ALTER ROLE %I WITH LOGIN PASSWORD %L', :'owner', :'owner_password') \gexec

SELECT format('CREATE ROLE %I', :'reader')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'reader') \gexec
SELECT format('ALTER ROLE %I WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD %L',
              :'reader', :'reader_password') \gexec
-- Extra guard: the reader's sessions start in read-only transactions.
SELECT format('ALTER ROLE %I SET default_transaction_read_only = on', :'reader') \gexec

-- Database, owned by the owner role.
SELECT format('CREATE DATABASE %I OWNER %I', :'app_db', :'owner')
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'app_db') \gexec
SELECT format('GRANT CONNECT ON DATABASE %I TO %I', :'app_db', :'reader') \gexec

-- Read access inside the database: existing tables, and tables the owner creates later.
\connect :"app_db"
SELECT format('GRANT USAGE ON SCHEMA public TO %I', :'reader') \gexec
SELECT format('GRANT SELECT ON ALL TABLES IN SCHEMA public TO %I', :'reader') \gexec
SELECT format('ALTER DEFAULT PRIVILEGES FOR ROLE %I IN SCHEMA public GRANT SELECT ON TABLES TO %I',
              :'owner', :'reader') \gexec
SQL

  echo "ensure_app_db: database '$APP_DB_NAME' ready (owner '$APP_DB_OWNER', read-only '$APP_DB_READER')"
}

ensure_app_db &

# Hand PID 1 to the official entrypoint so Postgres gets signals directly.
exec docker-entrypoint.sh "$@"
