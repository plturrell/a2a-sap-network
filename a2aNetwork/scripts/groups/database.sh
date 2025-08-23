#!/bin/bash
# Consolidated database operations script for A2A Network

set -e

COMMAND=${1:-"help"}
ARGS="${@:2}"

case $COMMAND in
  "deploy")
    echo "Deploying database to SQLite..."
    cds deploy --to sqlite $ARGS
    ;;
  "migrate")
    echo "Migrating database to HANA..."
    cds deploy --to hana $ARGS
    ;;
  "seed")
    echo "Seeding database with initial data..."
    node scripts/seed-data.js $ARGS
    ;;
  "reset")
    echo "Resetting database..."
    rm -f data/*.db
    cds deploy --to sqlite
    node scripts/seed-data.js $ARGS
    ;;
  "backup")
    echo "Creating database backup..."
    node scripts/create-backup.js $ARGS
    ;;
  "restore")
    echo "Restoring database from backup..."
    node scripts/restore-backup.js $ARGS
    ;;
  "setup")
    echo "Setting up fresh database..."
    echo "1. Deploying schema..."
    cds deploy --to sqlite
    echo "2. Seeding data..."
    node scripts/seed-data.js
    echo "3. Creating initial backup..."
    node scripts/create-backup.js $ARGS
    ;;
  *)
    echo "Available database commands:"
    echo "  deploy   - Deploy to SQLite"
    echo "  migrate  - Migrate to HANA"
    echo "  seed     - Seed with initial data"
    echo "  reset    - Reset and reseed database"
    echo "  backup   - Create backup"
    echo "  restore  - Restore from backup"
    echo "  setup    - Full database setup"
    echo "Usage: npm run db [command] [options]"
    ;;
esac