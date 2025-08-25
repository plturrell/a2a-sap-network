# Testing Scripts

This directory contains scripts for testing and verification of the A2A Platform.

## Scripts Overview

- **`verify-18-steps.sh`** - Comprehensive 18-step verification of the entire platform

## 18-Step Verification

The verification script validates:

1. **Pre-flight Checks** - Basic file existence
2. **Environment Setup** - Directory structure
3. **Infrastructure Services** - Redis, DB configs
4. **Blockchain Services** - Smart contracts
5. **Core Services** - Network, Frontend
6. **Trust System** - Security components
7. **Agent Services** - All 18 agents
   - Agent 0-5: Core agents
   - Agent 6-11: Extended agents
   - Agent 12-17: Specialized agents
8. **MCP Servers** - Model Context Protocol
9. **API Gateway** - Routing and load balancing
10. **Frontend UI** - User interface
11. **Authentication** - Security layer
12. **Database** - Data persistence
13. **Caching** - Performance layer
14. **Message Queue** - Async processing
15. **Monitoring** - Observability
16. **Logging** - Centralized logs
17. **Health Checks** - Service health
18. **Integration Tests** - End-to-end validation

## Usage

### Run Full Verification
```bash
./verify-18-steps.sh
```

### Output Example
```
🔍 A2A System 18-Step Verification
==================================

Step 1/18: Pre-flight Checks... ✅ PASS
Step 2/18: Environment Setup... ✅ PASS
Step 3/18: Infrastructure Services... ✅ PASS
...
Step 18/18: Integration Tests... ✅ PASS

📊 Verification Summary
======================
Total Steps: 18
Passed: 18
Failed: 0

✅ System verification PASSED!
```

## Integration with CI/CD

The verification script is used in:
- GitHub Actions workflows
- Docker health checks
- Deployment validation
- Pre-release testing

### CI Mode
```bash
# Run in CI mode (fails fast)
CI=true ./verify-18-steps.sh
```

## Custom Verification

Add custom checks by extending the script:

```bash
# Add to verify-18-steps.sh
verify_step 19 "Custom Check" "test -f /path/to/file"
```

## Debugging Failed Steps

If a step fails:

1. Check the specific component:
   ```bash
   # For agent issues
   curl http://localhost:8000/health
   
   # For service issues
   docker-compose ps
   ```

2. Review logs:
   ```bash
   # Check specific service logs
   tail -f logs/agent0.log
   ```

3. Run individual checks:
   ```bash
   # Test specific agent
   python -m a2aAgents.backend.main --agent=0
   ```

## Future Testing Scripts

Planned additions:
- `integration-test.sh` - API integration tests
- `load-test.sh` - Performance testing
- `security-scan.sh` - Security validation
- `smoke-test.sh` - Quick health checks