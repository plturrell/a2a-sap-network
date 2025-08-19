# A2A Network SAP Fiori Launchpad

## Overview
Production-ready SAP Fiori Launchpad for the A2A Network Management Platform. Built following SAP enterprise standards with full observability, personalization, caching, and monitoring capabilities.

## Features

### ✅ Enterprise-Grade Implementation
- **SAP Fiori Compliance**: 100% adherent to SAP design guidelines and standards
- **Dual Environment Support**: Local development (SQLite + JWT) and BTP production (HANA + XSUAA)
- **Real Database Persistence**: User preferences, tile configurations, and group settings
- **Advanced Caching**: Redis with automatic fallback to in-memory cache
- **Complete Observability**: OpenTelemetry with Jaeger/Prometheus integration
- **SAP Cloud ALM Integration**: Application lifecycle management and monitoring

### 🔐 Security & Authentication
- **XSUAA Integration**: Full SAP BTP authentication for production
- **JWT Authentication**: Secure local development authentication
- **Input Validation**: Comprehensive request sanitization and validation
- **Security Headers**: Helmet.js security middleware
- **Rate Limiting**: Protection against abuse and DoS attacks

### 📊 Monitoring & Analytics
- **Health Endpoints**: Comprehensive system health monitoring
- **Metrics Collection**: Custom business and technical metrics
- **Distributed Tracing**: Full request tracing with OpenTelemetry
- **Performance Analytics**: Response time and load monitoring
- **Error Reporting**: Centralized error collection and analysis

### 🎨 Personalization
- **Tile Customization**: Position, size, visibility, and refresh intervals
- **User Preferences**: Themes, layout modes, and personal settings
- **Group Management**: Custom group arrangements and titles
- **Import/Export**: Configuration backup and restore
- **Usage Analytics**: Tile usage patterns and optimization insights

## Quick Start

### Prerequisites
- Node.js 18+
- npm 9+
- SQLite (local) or SAP HANA (BTP)
- Redis (optional, has fallback)

### Local Development
```bash
# Install dependencies
npm install

# Set up environment
cp .env.example .env

# Start development server
npm run dev

# Run tests
npm test

# Run integration tests
npm run test:integration

# Check health
npm run health-check
```

### Production Deployment (SAP BTP)
```bash
# Configure BTP environment
cf login
cf target -o <org> -s <space>

# Deploy to Cloud Foundry
npm run deploy:btp

# Validate deployment
cf logs a2a-network-launchpad --recent
```

## Architecture

### Environment Detection
The application automatically detects its environment and configures accordingly:

- **Local Development**: SQLite database, JWT auth, console logging
- **SAP BTP Production**: HANA database, XSUAA auth, cloud logging

### Key Components

#### 1. Production Launchpad Server (`production-launchpad-server.js`)
Main application server with environment-aware configuration.

#### 2. Personalization Service (`personalization/tilePersonalization.js`)
Database-persistent user preferences and tile configurations.

#### 3. Caching Layer (`caching/redisCache.js`)
Redis-based distributed caching with in-memory fallback.

#### 4. Observability (`observability/telemetry.js`)
OpenTelemetry integration for metrics and tracing.

#### 5. SAP Cloud ALM (`monitoring/sapCloudALM.js`)
Enterprise monitoring and application lifecycle management.

### Database Schema

#### User Tile Configuration
```sql
CREATE TABLE user_tile_config (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    tile_id TEXT NOT NULL,
    group_id TEXT NOT NULL,
    position INTEGER DEFAULT 0,
    is_visible BOOLEAN DEFAULT 1,
    size TEXT DEFAULT '1x1',
    custom_title TEXT,
    refresh_interval INTEGER DEFAULT 30,
    settings TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## API Reference

### Authentication
```http
POST /auth/login
Content-Type: application/json

{
  "username": "user@company.com",
  "password": "password"
}
```

### Tile Data
```http
GET /api/tiles/data
Authorization: Bearer <token>
```

### Personalization
```http
POST /api/personalization/tiles/{tileId}
Authorization: Bearer <token>
Content-Type: application/json

{
  "userId": "user-123",
  "groupId": "group-456",
  "config": {
    "position": 2,
    "size": "2x2",
    "isVisible": true
  }
}
```

### Health & Monitoring
```http
GET /health
GET /metrics
GET /api/cache/stats
```

## Testing

### Comprehensive Test Suite
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end functionality testing  
- **Performance Tests**: Load and response time validation
- **Security Tests**: Input validation and auth testing
- **SAP Compliance Tests**: Standards adherence validation

### Running Tests
```bash
# All tests
npm test

# Integration tests only
npm run test:integration

# Watch mode
npm run test:watch

# Coverage report
npm run coverage
```

## Monitoring & Observability

### Health Monitoring
- **Application Health**: `/health` endpoint with detailed service status
- **Database Health**: Connection and query performance monitoring
- **Cache Health**: Redis availability and fallback status
- **Authentication Health**: Token validation and user session monitoring

### Metrics Collection
- **Business Metrics**: User activity, tile interactions, agent operations
- **Technical Metrics**: Response times, error rates, cache hit ratios
- **System Metrics**: Memory usage, CPU utilization, active connections

### Distributed Tracing
- **Request Tracing**: Full request lifecycle tracking
- **Database Queries**: Query performance and optimization
- **Cache Operations**: Hit/miss patterns and performance
- **External APIs**: Third-party service call monitoring

## Configuration

### Environment Variables

#### Database
- `DATABASE_TYPE`: sqlite | hana
- `SQLITE_DB_PATH`: Path to SQLite database file
- `HANA_HOST`, `HANA_PORT`, `HANA_DATABASE`: HANA connection details

#### Authentication  
- `BTP_ENVIRONMENT`: true | false
- `ENABLE_XSUAA_VALIDATION`: true | false
- `JWT_SECRET`: JWT signing secret

#### Caching
- `REDIS_URL`: Redis connection string
- `ENABLE_CACHE`: true | false

#### Monitoring
- `ENABLE_METRICS`: true | false
- `JAEGER_ENDPOINT`: Jaeger tracing endpoint
- `SAP_CLOUD_ALM_URL`: SAP Cloud ALM service URL

## Performance Optimization

### Caching Strategy
- **API Responses**: 30-second TTL for tile data
- **User Sessions**: 1-hour TTL for authentication
- **Static Assets**: Browser caching with ETags
- **Database Queries**: Query result caching with smart invalidation

### Database Optimization
- **Connection Pooling**: Optimized for both SQLite and HANA
- **Query Optimization**: Indexed fields and efficient queries
- **Batch Operations**: Bulk inserts and updates for performance

### Resource Management
- **Memory Management**: Automatic cleanup and garbage collection
- **Connection Limits**: Proper connection pool management
- **Graceful Shutdown**: Clean resource cleanup on termination

## Security

### Authentication & Authorization
- **Token-Based Auth**: JWT for local, XSUAA for BTP
- **Role-Based Access**: User roles and permissions
- **Session Management**: Secure session handling

### Input Validation & Sanitization
- **Request Validation**: All inputs validated and sanitized
- **SQL Injection Prevention**: Parameterized queries only
- **XSS Protection**: Content Security Policy headers

### Security Headers
- **Helmet.js**: Comprehensive security header management
- **CORS Configuration**: Proper cross-origin resource sharing
- **Rate Limiting**: Protection against abuse

## Troubleshooting

### Common Issues

#### Database Connection Errors
```bash
# Check database configuration
npm run validate-config

# Test database connection
node -e "require('./production-launchpad-server.js')"
```

#### Cache Issues
```bash
# Check Redis status
redis-cli ping

# Clear cache
curl -X DELETE http://localhost:4004/api/cache/clear
```

#### Authentication Problems
```bash
# Verify JWT configuration
echo $JWT_SECRET

# Test authentication
curl -X POST http://localhost:4004/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test@example.com","password":"test123"}'
```

### Logs and Debugging
```bash
# View application logs
tail -f logs/application.log

# Check health status
curl http://localhost:4004/health

# View metrics
curl http://localhost:4004/metrics
```

## Contributing

1. Follow SAP development standards
2. Add tests for new features
3. Update documentation
4. Run linting and tests before committing

## License
MIT License - see LICENSE file for details

## Support
For support and questions, please create an issue in the GitHub repository.