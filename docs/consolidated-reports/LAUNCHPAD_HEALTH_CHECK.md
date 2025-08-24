# Launchpad Health Check System

This document describes the comprehensive health check system for the A2A Network Launchpad to ensure it loads properly with real tiles and no blank screens.

## Overview

The health check system provides multiple layers of validation:

1. **Startup Health Check** - Runs automatically when server starts
2. **Comprehensive Validation** - Full visual and functional testing
3. **Integration Tests** - Automated browser-based testing
4. **Quick Readiness Test** - Fast verification of basic functionality

## Health Check Components

### 1. Startup Health Check (`scripts/startup-health-check.js`)

**Purpose**: Validates launchpad functionality during server startup

**When it runs**: Automatically during server startup (integrated into `srv/server.js`)

**What it checks**:
- Server responsiveness
- API endpoint availability
- Tile data structure validity
- Component health status

**Usage**:
```bash
npm run health:startup
```

**Integration**: The health check is automatically executed during server startup and logs results to the console.

### 2. Comprehensive Validation (`scripts/validate-launchpad.js`)

**Purpose**: Full validation including visual rendering using Puppeteer

**When to use**: 
- Before production deployments
- After major changes
- Troubleshooting rendering issues

**What it checks**:
- Server health
- API endpoints
- UI5 resource availability
- Visual tile rendering
- Error detection
- Screenshot capture

**Usage**:
```bash
npm run validate:launchpad
```

**Exit codes**:
- `0`: All tests passed with real data
- `1`: Tests passed but using fallback data
- `2`: Critical failures detected
- `3`: Validation error

### 3. Integration Tests (`test/launchpad-integration.test.js`)

**Purpose**: Automated browser testing for CI/CD pipelines

**Test framework**: Mocha + Chai + Puppeteer

**What it tests**:
- Launchpad loading without errors
- SAP UI5 framework initialization
- Tile container rendering
- Exact tile count (6 tiles)
- Tile content validation
- API data fetching
- Visual consistency

**Usage**:
```bash
npm test -- test/launchpad-integration.test.js
```

### 4. Quick Readiness Test (`scripts/test-launchpad-ready.js`)

**Purpose**: Fast verification for development

**When to use**:
- Quick health checks during development
- CI/CD pipeline smoke tests
- Automated monitoring

**Usage**:
```bash
node scripts/test-launchpad-ready.js
```

## Health Check Endpoints

### `/api/v1/launchpad/health`

Comprehensive health check endpoint that returns:

```json
{
  "timestamp": "2025-08-22T07:45:00.000Z",
  "status": "healthy|degraded|error",
  "components": {
    "ui5_resources": { "status": "healthy" },
    "shell_config": { "status": "healthy" },
    "api_endpoints": { "status": "healthy", "details": [...] },
    "tile_data": { "status": "warning", "details": {...} },
    "websocket": { "status": "healthy" }
  },
  "tiles_loaded": true,
  "real_data_available": false,
  "fallback_mode": true,
  "recommendations": [
    "Start agent services on ports 8000-8015 for real data"
  ]
}
```

## Status Definitions

### Component Status
- **healthy**: Component functioning correctly
- **warning**: Component working but with limitations (e.g., no real data)
- **error**: Component not functioning

### Overall Status
- **healthy**: All critical components working, real data available
- **degraded**: Critical components working, but with warnings
- **error**: One or more critical components failing

### Data Status
- **Real data**: Agent services running, providing live metrics
- **Fallback mode**: Using default/cached data when services unavailable

## Troubleshooting

### Common Issues

1. **No tiles rendered**
   - Check: UI5 resources loading
   - Check: View/Controller initialization
   - Check: Model binding

2. **Blank screen**
   - Check: JavaScript errors in browser console
   - Check: SAP shell bootstrap completion
   - Check: Authentication issues

3. **Fallback data only**
   - Check: Agent services running on ports 8000-8015
   - Check: API endpoint authentication
   - Check: Network connectivity

### Debug Commands

```bash
# Quick health check
npm run health:startup

# Full validation with screenshots
npm run validate:launchpad

# Run integration tests
npm test -- test/launchpad-integration.test.js

# Check specific endpoint
curl http://localhost:4004/api/v1/launchpad/health | jq '.'
```

## Integration with Startup Process

The health check is automatically integrated into the server startup process:

1. Server starts and initializes all services
2. Health check runs after 2-second delay
3. Results are logged to console
4. Server continues regardless of health check results
5. Health status available via `/api/v1/launchpad/health` endpoint

## Best Practices

1. **Development**:
   - Run `node scripts/test-launchpad-ready.js` after code changes
   - Check health endpoint regularly: `/api/v1/launchpad/health`

2. **Testing**:
   - Include integration tests in CI/CD pipeline
   - Use `npm run validate:launchpad` before deployments

3. **Production**:
   - Monitor health endpoint for degradation
   - Set up alerts for `error` status
   - Regular validation runs

4. **Debugging**:
   - Check browser console for JavaScript errors
   - Use comprehensive validation for detailed diagnostics
   - Review screenshots from validation runs

## Configuration

Health check behavior can be configured via environment variables:

- `HEADLESS=false`: Run Puppeteer in visible mode for debugging
- `TEST_URL=http://custom:port`: Use custom server URL for testing

## Dependencies

Required packages for full health check functionality:

- `puppeteer`: Browser automation for visual validation
- `chalk`: Console output formatting
- `mocha`/`chai`: Test framework (for integration tests)

Install if missing:
```bash
npm install puppeteer chalk mocha chai --save-dev
```