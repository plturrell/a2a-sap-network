#!/bin/bash
# Database migration script for Fly.io deployments

set -e

APP_NAME="${FLY_APP_NAME:-a2a-platform}"
COMMAND="${1:-status}"

echo "🗄️  Fly.io Database Migration Tool"
echo "================================="
echo "App: $APP_NAME"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if we're in Fly.io environment
check_fly_env() {
    if [ -z "$FLY_APP_NAME" ] && [ "$APP_NAME" = "a2a-platform" ]; then
        echo -e "${YELLOW}⚠️  Warning: Not in Fly.io environment${NC}"
        echo "Set FLY_APP_NAME or run within Fly.io"
    fi
}

# Function to run command in Fly.io
run_in_fly() {
    local cmd=$1
    echo -e "${BLUE}Running: $cmd${NC}"
    
    if [ -n "$FLY_APP_NAME" ]; then
        # We're already in Fly.io, run directly
        eval "$cmd"
    else
        # Run via SSH
        flyctl ssh console --app "$APP_NAME" --command "$cmd"
    fi
}

# Function to get database URL
get_database_url() {
    if [ -n "$DATABASE_URL" ]; then
        echo "$DATABASE_URL"
    else
        # Try to get from Fly.io secrets
        flyctl secrets list --app "$APP_NAME" --json 2>/dev/null | \
            jq -r '.[] | select(.Name == "DATABASE_URL") | .Name' > /dev/null && \
            echo "Using DATABASE_URL from Fly.io secrets" >&2
        echo ""
    fi
}

# Main commands
case "$COMMAND" in
    status)
        echo -e "${BLUE}📊 Migration Status${NC}"
        echo "=================="
        
        # Check if alembic is available
        if run_in_fly "cd /app && python -c 'import alembic' 2>/dev/null"; then
            echo -e "${GREEN}✅ Alembic is installed${NC}"
            
            # Get current revision
            echo ""
            echo "Current database revision:"
            run_in_fly "cd /app && alembic current" || echo "No migrations found"
            
            # Show migration history
            echo ""
            echo "Migration history:"
            run_in_fly "cd /app && alembic history --verbose" || echo "No history available"
        else
            echo -e "${RED}❌ Alembic not found${NC}"
            echo "Please ensure alembic is in requirements.txt"
        fi
        ;;
        
    init)
        echo -e "${BLUE}🔧 Initializing Migrations${NC}"
        echo "========================"
        
        # Initialize alembic if not already done
        if run_in_fly "test -f /app/alembic.ini"; then
            echo -e "${YELLOW}⚠️  Alembic already initialized${NC}"
        else
            run_in_fly "cd /app && alembic init migrations"
            echo -e "${GREEN}✅ Alembic initialized${NC}"
        fi
        
        # Create initial migration
        echo ""
        echo "Creating initial migration..."
        run_in_fly "cd /app && alembic revision --autogenerate -m 'Initial migration'"
        echo -e "${GREEN}✅ Initial migration created${NC}"
        ;;
        
    migrate)
        echo -e "${BLUE}🚀 Running Migrations${NC}"
        echo "===================="
        
        # Backup warning
        echo -e "${YELLOW}⚠️  Warning: This will modify the database${NC}"
        echo "Ensure you have a backup before proceeding!"
        echo ""
        
        if [ "$2" != "--force" ]; then
            read -p "Continue? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Migration cancelled"
                exit 0
            fi
        fi
        
        # Run migrations
        echo "Running database migrations..."
        if run_in_fly "cd /app && alembic upgrade head"; then
            echo -e "${GREEN}✅ Migrations completed successfully${NC}"
        else
            echo -e "${RED}❌ Migration failed${NC}"
            exit 1
        fi
        ;;
        
    rollback)
        echo -e "${BLUE}⏪ Rolling Back Migration${NC}"
        echo "========================"
        
        # Get target revision
        target="${2:--1}"
        
        echo -e "${YELLOW}⚠️  Warning: This will rollback the database${NC}"
        echo "Target: $target"
        echo ""
        
        read -p "Continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Rollback cancelled"
            exit 0
        fi
        
        # Perform rollback
        echo "Rolling back migration..."
        if run_in_fly "cd /app && alembic downgrade $target"; then
            echo -e "${GREEN}✅ Rollback completed${NC}"
        else
            echo -e "${RED}❌ Rollback failed${NC}"
            exit 1
        fi
        ;;
        
    create)
        echo -e "${BLUE}📝 Creating New Migration${NC}"
        echo "========================"
        
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Error: Migration name required${NC}"
            echo "Usage: $0 create 'migration name'"
            exit 1
        fi
        
        migration_name="$2"
        echo "Creating migration: $migration_name"
        
        if run_in_fly "cd /app && alembic revision --autogenerate -m '$migration_name'"; then
            echo -e "${GREEN}✅ Migration created${NC}"
            echo ""
            echo "Next steps:"
            echo "1. Review the generated migration file"
            echo "2. Test in staging environment"
            echo "3. Run: $0 migrate"
        else
            echo -e "${RED}❌ Failed to create migration${NC}"
            exit 1
        fi
        ;;
        
    backup)
        echo -e "${BLUE}💾 Database Backup${NC}"
        echo "================="
        
        timestamp=$(date +%Y%m%d-%H%M%S)
        backup_file="a2a-backup-$timestamp.sql"
        
        echo "Creating backup: $backup_file"
        
        # Get database URL
        db_url=$(get_database_url)
        if [ -z "$db_url" ]; then
            echo -e "${RED}❌ Error: DATABASE_URL not found${NC}"
            exit 1
        fi
        
        # Create backup
        if run_in_fly "cd /app && pg_dump $db_url > /tmp/$backup_file"; then
            echo -e "${GREEN}✅ Backup created${NC}"
            
            # Download backup
            echo "Downloading backup..."
            flyctl ssh sftp get "/tmp/$backup_file" --app "$APP_NAME"
            echo -e "${GREEN}✅ Backup downloaded: $backup_file${NC}"
            
            # Cleanup remote file
            run_in_fly "rm /tmp/$backup_file"
        else
            echo -e "${RED}❌ Backup failed${NC}"
            exit 1
        fi
        ;;
        
    validate)
        echo -e "${BLUE}🔍 Validating Database${NC}"
        echo "====================="
        
        # Check database connection
        echo -n "Checking database connection... "
        if run_in_fly "cd /app && python -c 'from app.core.database import engine; engine.connect()' 2>/dev/null"; then
            echo -e "${GREEN}✅ Connected${NC}"
        else
            echo -e "${RED}❌ Connection failed${NC}"
            exit 1
        fi
        
        # Check tables
        echo ""
        echo "Database tables:"
        run_in_fly "cd /app && python -c '
from app.core.database import engine
from sqlalchemy import inspect
inspector = inspect(engine)
for table in inspector.get_table_names():
    print(f\"  - {table}\")
'" || echo "Unable to list tables"
        
        # Check migration status
        echo ""
        echo "Migration status:"
        run_in_fly "cd /app && alembic current" || echo "No migrations"
        ;;
        
    setup-ci)
        echo -e "${BLUE}🔧 Setting up CI/CD Migration${NC}"
        echo "============================="
        
        # Create GitHub Action for migrations
        cat > .github/workflows/db-migrate.yml << 'EOF'
name: Database Migration

on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to migrate'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production
      migration_type:
        description: 'Migration type'
        required: true
        default: 'migrate'
        type: choice
        options:
          - migrate
          - rollback
          - status

jobs:
  migrate:
    runs-on: ubuntu-latest
    environment: ${{ github.event.inputs.environment }}
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Fly.io
        uses: superfly/flyctl-actions/setup-flyctl@master
      
      - name: Run migration
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
          FLY_APP_NAME: ${{ vars.FLY_APP_NAME }}
        run: |
          ./scripts/fly-db-migrate.sh ${{ github.event.inputs.migration_type }}
EOF
        
        echo -e "${GREEN}✅ CI/CD workflow created${NC}"
        echo "Commit and push to enable automated migrations"
        ;;
        
    *)
        echo "Usage: $0 {status|init|migrate|rollback|create|backup|validate|setup-ci} [args]"
        echo ""
        echo "Commands:"
        echo "  status              - Show current migration status"
        echo "  init               - Initialize migrations"
        echo "  migrate [--force]  - Run pending migrations"
        echo "  rollback [target]  - Rollback to specific revision"
        echo "  create 'name'      - Create new migration"
        echo "  backup             - Create database backup"
        echo "  validate           - Validate database connection"
        echo "  setup-ci           - Set up CI/CD for migrations"
        echo ""
        echo "Examples:"
        echo "  $0 status"
        echo "  $0 create 'add user table'"
        echo "  $0 migrate --force"
        echo "  $0 rollback -1"
        exit 1
        ;;
esac