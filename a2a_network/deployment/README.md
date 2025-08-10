# A2A Network - Deployment Guide

## Environment Configuration

The A2A Network application uses environment-based configuration to seamlessly transition between local development and SAP BTP production environments.

### Key Environment Variable: `ENABLE_XSUAA_VALIDATION`

This critical variable controls the authentication mode:

- **`false`** (Local Development): Uses development JWT decoding with mock user data
- **`true`** (BTP Production): Uses full XSUAA validation with SAP security

### Local Development Setup

1. **Development Environment** (`.env`):
   ```bash
   NODE_ENV=development
   ENABLE_XSUAA_VALIDATION=false
   ```

2. **Features in Development Mode**:
   - JWT tokens are decoded but not validated (no signature verification)
   - Mock user data is generated from token payload
   - Full admin/developer access granted
   - Warning messages logged for non-production usage

### BTP Production Deployment

1. **Copy Production Configuration**:
   ```bash
   cp deployment/btp-production.env .env
   ```

2. **Production Environment** (`.env`):
   ```bash
   NODE_ENV=production
   ENABLE_XSUAA_VALIDATION=true
   ```

3. **Features in Production Mode**:
   - Full XSUAA JWT signature validation
   - Real user data from SAP BTP security context
   - Role-based access control enforced
   - Scope-based permissions validated

### BTP Service Bindings Required

The application requires these BTP services:

1. **XSUAA Service**:
   ```json
   {
     "name": "a2a-network-xsuaa",
     "service": "xsuaa",
     "service-plan": "application"
   }
   ```

2. **HANA Cloud**:
   ```json
   {
     "name": "a2a-network-hana",
     "service": "hana-cloud",
     "service-plan": "hana"
   }
   ```

### Security Model

The application implements SAP's security best practices:

- **Scopes**: `Admin`, `Developer`, `User`, `ServiceAccount`
- **Role Templates**: Granular permission sets
- **Role Collections**: User assignment groups
- **Multi-tenancy**: Tenant isolation support

### Deployment Checklist

Before deploying to BTP:

- [ ] Set `ENABLE_XSUAA_VALIDATION=true`
- [ ] Configure production XSUAA service binding
- [ ] Update HANA connection credentials
- [ ] Generate new security secrets
- [ ] Configure production CORS origins
- [ ] Deploy xs-security.json configuration
- [ ] Test authentication with real BTP users

### Monitoring

The application provides:
- `/health` endpoint for service monitoring
- Detailed logging for authentication events
- Role and scope validation logging
- Database connectivity status

For production deployment, monitor the logs for:
- XSUAA configuration loading success
- JWT validation success/failure rates
- Role-based access denials
- Database connection health