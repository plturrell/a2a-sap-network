# A2A Network Security Documentation

## Security Policy

We take the security of A2A Network contracts and applications seriously and appreciate responsible disclosures.

### Supported Versions

| Version | Supported |
|---------|-----------|
| `main`  | ✅ |

### Reporting a Vulnerability

1. **Do not create a public issue or PR.**
2. Email **security@a2a.network** with the following:
   - Detailed description of the vulnerability and potential impact.
   - Steps to reproduce (PoC, scripts, or transaction traces).
   - Your contact information and PGP key (optional).
3. We will acknowledge receipt within **48 hours** and provide a timeline for triage and remediation.
4. We aim to release a fix within **14 days** of confirmation, followed by a public disclosure.

### Bug Bounty

Significant vulnerabilities may be eligible for a bounty in ETH or USDC, determined by severity and exploitability.

### Scope

- Contracts in `src/` directory on the `main` branch
- Deployment scripts
- SAP CAP application services and middleware

**Out of scope:** Low-impact issues (e.g. gas optimizations), third-party dependencies, or already-fixed bugs.

## Security Implementation Status

### ✅ Implemented Security Enhancements

1. **CORS Protection** - Replaced wildcard origin with environment-based whitelist
2. **Rate Limiting** - Multi-tier rate limiting for different endpoint types
3. **Security Headers** - Comprehensive headers via Helmet.js including CSP
4. **Authentication** - JWT and API key validation with role-based access
5. **Input Sanitization** - Request cleaning and validation middleware
6. **Session Security** - Secure cookie configuration
7. **Environment Config** - Secure secrets management with templates

### Security Components

#### 1. Security Middleware (`srv/middleware/security.js`)
- **CORS Configuration**: Environment-based origin whitelist
- **Rate Limiting**: Different limits for auth, read, write, and blockchain operations
- **Helmet.js**: Comprehensive security headers
- **Request Sanitization**: Input validation and size limits

#### 2. Authentication Middleware (`srv/middleware/auth.js`)
- **JWT Validation**: Token-based authentication
- **API Key Support**: Service-to-service authentication
- **Role-Based Access**: Middleware for role requirements
- **Session Management**: Secure session configuration

## Configuration

### Rate Limits

| Endpoint Type | Limit | Window |
|--------------|-------|---------|
| General API | 100 requests | 15 minutes |
| Authentication | 5 requests | 15 minutes |
| Read Operations | 200 requests | 15 minutes |
| Write Operations | 50 requests | 15 minutes |
| Blockchain | 10 requests | 1 hour |

### Security Headers Applied

- **Content-Security-Policy**: Restrictive policy with SAP UI5 exceptions
- **Strict-Transport-Security**: HSTS with 1-year max-age
- **X-Frame-Options**: DENY
- **X-Content-Type-Options**: nosniff
- **X-XSS-Protection**: 1; mode=block
- **Referrer-Policy**: strict-origin-when-cross-origin

### CORS Configuration

Production:
```javascript
ALLOWED_ORIGINS=https://a2a-network.cfapps.eu10.hana.ondemand.com
```

Development:
```javascript
ALLOWED_ORIGINS=http://localhost:4004,http://localhost:3000
```

## Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your secure values
```

### 3. Run Security Audit
```bash
node scripts/security-audit.js
```

### 4. Start Server
```bash
npm start
```

## Testing Security

### Manual Testing

1. **Test CORS**:
```bash
curl -H "Origin: http://evil.com" http://localhost:4004/api/v1/Agents
# Should be blocked
```

2. **Test Rate Limiting**:
```bash
for i in {1..10}; do curl http://localhost:4004/api/v1/auth/login; done
# Should block after 5 requests
```

3. **Test Security Headers**:
```bash
curl -I http://localhost:4004/health
# Should show security headers
```

### Automated Testing

Run the security audit:
```bash
node scripts/security-audit.js
```

Expected output:
```
✅ Security middleware is applied
✅ Authentication middleware is applied
✅ CORS properly configured
✅ Rate limiting is configured
✅ Helmet security headers configured
✅ Input sanitization middleware present
```

## Security Checklist

- [x] CORS policy restricts origins
- [x] Rate limiting prevents abuse
- [x] Security headers protect against common attacks
- [x] Authentication required for protected endpoints
- [x] Input sanitization prevents injection
- [x] Session cookies are secure
- [x] Environment template provided
- [x] Security audit script available

## Production Deployment Security

### Before Production:
- Update all secrets in `.env`
- Set `NODE_ENV=production`
- Configure real XSUAA service
- Update ALLOWED_ORIGINS
- Enable SSL/TLS

### Secrets Management:
- Never commit `.env` files
- Use strong, random secrets
- Rotate secrets regularly
- Use SAP BTP secrets service in production

### Monitoring:
- Monitor rate limit hits
- Track authentication failures
- Log security events
- Set up alerts for security violations

## References

- [Ethereum Smart Contract Best Practices](https://consensys.github.io/smart-contract-best-practices/)
- [SWC Registry](https://swcregistry.io)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [SAP Security Guidelines](https://help.sap.com/docs/SAP_HANA_PLATFORM/b3ee5778bc2e4a089d3299b82ec762a7/c511ee70ba83405580f8c8b4f7c0e7b7.html)

## Support

For security questions or concerns:
- Review this documentation for implementation details
- Check logs in development for security events
- Use security audit script for validation
- Email security@a2a.network for vulnerability reports