# Production Deployment Guide

## Files and Directories to Exclude

When deploying to production, exclude the following:

### Directories to exclude:
- `scripts/verification/`
- `scripts/diagnostics/`
- `scripts/temp/`
- `scripts/test/`
- `docs/testing/`
- `docs/implementation/` (keep only essential docs)
- `logs/` (create fresh in production)
- `data/test_results/`
- `.pytest_cache/`
- `__pycache__/`
- `*.egg-info/`
- `.git/`
- `node_modules/` (for developer portal, install fresh)

### Files to exclude:
- `*.log`
- `.env*` (except .env.example)
- `*test*.py` (in root directory)
- `*debug*.py`
- `*temp*.py`
- `.gitignore`
- `.pre-commit-config.yaml`
- `pytest.ini` (only needed for development)

### Production Setup:
1. Use environment variables for all secrets
2. Install only production dependencies: `pip install -e .`
3. For developer portal: `cd app/a2a/developer_portal/cap && npm ci --production`
4. Set appropriate file permissions
5. Configure log rotation
6. Set up monitoring endpoints

### Security Checklist:
- [ ] Remove all .env files
- [ ] Verify no hardcoded secrets
- [ ] Set production configuration
- [ ] Enable HTTPS only
- [ ] Configure firewall rules
- [ ] Set up rate limiting
- [ ] Enable security headers
