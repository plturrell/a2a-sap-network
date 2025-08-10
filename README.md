# A2A Developer Portal

A comprehensive SAP BTP enterprise application for agent-to-agent development and deployment management.

## 🚀 Features

### Sprint 1: Core Foundation
- ✅ UI5 application with modern Fiori design
- ✅ Project management and file handling
- ✅ Agent builder with configuration management
- ✅ BPMN workflow designer integration
- ✅ Testing framework and validation tools
- ✅ Deployment pipeline management

### Sprint 2: SAP BTP Integration
- ✅ SAP Application Router (xs-app.json)
- ✅ XSUAA authentication service
- ✅ Role-based access control (RBAC)
- ✅ Multi-target application deployment (mta.yaml)
- ✅ SAP Destination Service integration
- ✅ Session management with Redis backend
- ✅ User profile management UI

### Sprint 3: Fiori Elements & Smart Controls
- ✅ SmartTable with advanced filtering and sorting
- ✅ SmartFilterBar for complex search operations
- ✅ Master-Detail pattern with flexible column layout
- ✅ Overview Page with KPI cards and dashboards
- ✅ Fiori Launchpad integration with semantic navigation
- ✅ Cross-application navigation service
- ✅ Variant management for user customization

### Sprint 4: Enterprise Features & CAP Migration
- ✅ SAP Cloud Application Programming Model (CAP) architecture
- ✅ Core Data Services (CDS) models for all business objects
- ✅ OData v4 services with draft support
- ✅ SAP Workflow integration for approval processes
- ✅ Application Logging Service with structured logging
- ✅ SAP Cloud ALM monitoring and health checks
- ✅ Comprehensive audit logging for compliance
- ✅ BTP deployment pipeline with CI/CD automation

## 🏗️ Architecture

### Backend (SAP CAP)
- **Node.js** runtime with SAP CAP framework
- **CDS Models** for data persistence and business logic
- **OData v4** services for UI5 consumption
- **SAP HANA Cloud** database
- **Redis** for session management
- **SAP BTP Services** integration (XSUAA, Workflow, Logging, etc.)

### Frontend (SAP UI5)
- **SAP UI5** with Fiori Elements
- **Smart Controls** for enterprise data management
- **Responsive Design** with flexible layouts
- **Variant Management** for user customization
- **Cross-App Navigation** with semantic objects

### DevOps & Monitoring
- **Jenkins CI/CD** pipeline
- **MTA** deployment to SAP BTP
- **SonarQube** code quality analysis
- **SAP Cloud ALM** monitoring
- **Audit Logging** for compliance

## 🛠️ Technology Stack

### Core Technologies
- **SAP CAP** (Cloud Application Programming Model)
- **SAP UI5** with Fiori Elements
- **Node.js** 18+
- **SAP HANA Cloud** database
- **Redis** for caching and sessions

### SAP BTP Services
- **XSUAA** (Authentication & Authorization)
- **SAP Workflow** (Business Process Management)
- **Application Logging Service**
- **SAP Cloud ALM** (Application Lifecycle Management)
- **Destination Service** (External Connectivity)
- **Audit Logging Service** (Compliance)

### Development Tools
- **SAP Business Application Studio**
- **Jenkins** (CI/CD)
- **SonarQube** (Code Quality)
- **OWASP Dependency Check** (Security)
- **Jest** (Unit Testing)

## 📁 Project Structure

```
a2a/
├── a2a_agents/backend/app/a2a/developer_portal/
│   ├── cap/                          # SAP CAP backend
│   │   ├── db/                       # CDS data models
│   │   ├── srv/                      # Service implementations
│   │   ├── .pipeline/                # CI/CD configuration
│   │   ├── package.json              # CAP dependencies
│   │   └── Jenkinsfile              # Jenkins pipeline
│   ├── sap_btp/                     # SAP BTP integrations
│   │   ├── auth_api.py              # Authentication API
│   │   ├── rbac_service.py          # Role-based access control
│   │   ├── session_service.py       # Session management
│   │   └── destination_service.py   # Destination service
│   ├── static/                      # UI5 application
│   │   ├── controller/              # UI5 controllers
│   │   ├── view/                    # UI5 views and fragments
│   │   ├── services/                # Frontend services
│   │   ├── launchpad/               # Fiori Launchpad config
│   │   ├── manifest.json            # App descriptor
│   │   └── Component.js             # Main component
│   ├── portal_server.py             # Main server
│   ├── xs-app.json                  # App Router configuration
│   ├── xs-security.json             # XSUAA configuration
│   └── mta.yaml                     # MTA descriptor
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ and npm
- SAP CAP Development Kit (`@sap/cds-dk`)
- Python 3.8+ (for legacy backend components)
- SAP BTP account with required services
- Redis server (for session management)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd a2a
   ```

2. **Install CAP dependencies**
   ```bash
   cd a2a_agents/backend/app/a2a/developer_portal/cap
   npm install
   ```

3. **Install Python dependencies**
   ```bash
   cd ../
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your SAP BTP service credentials
   ```

### Development

1. **Start CAP development server**
   ```bash
   cd cap/
   npm run watch
   ```

2. **Start UI5 application**
   ```bash
   # The UI5 app is served by the CAP server at http://localhost:4004
   ```

3. **Access the application**
   - Main Portal: http://localhost:4004
   - Fiori Launchpad: http://localhost:4004/launchpad.html
   - OData Services: http://localhost:4004/api/v4/portal

### Testing

```bash
# Unit tests
npm run test

# Integration tests
npm run test:integration

# Code coverage
npm run test:coverage

# Lint code
npm run lint
```

### Deployment

1. **Build MTA archive**
   ```bash
   mbt build
   ```

2. **Deploy to SAP BTP**
   ```bash
   cf deploy mta_archives/a2a-developer-portal_1.0.0.mtar
   ```

## 🔧 Configuration

### SAP BTP Services
Configure the following services in your SAP BTP subaccount:
- XSUAA (Authentication)
- SAP HANA Cloud (Database)
- Application Logging
- SAP Workflow
- Destination Service
- Audit Logging

### Environment Variables
Key environment variables (see `.env.example`):
- `VCAP_SERVICES` - SAP BTP service bindings
- `NODE_ENV` - Environment (development/production)
- `LOG_LEVEL` - Logging level
- `REDIS_URL` - Redis connection string

## 📊 Monitoring & Observability

### Application Logging
- Structured JSON logging with correlation IDs
- Performance metrics and business events
- Integration with SAP Application Logging Service

### Health Monitoring
- Database connectivity checks
- Memory and CPU usage monitoring
- External service health validation
- SAP Cloud ALM integration

### Audit Logging
- Comprehensive audit trails for all business actions
- Data access and modification logging
- Authentication and authorization events
- Compliance reporting and retention policies

## 🔒 Security

### Authentication & Authorization
- SAP XSUAA integration with OAuth2/JWT
- Role-based access control (RBAC)
- Session management with secure tokens
- Multi-factor authentication support

### Data Protection
- Sensitive data sanitization in logs
- Encryption at rest and in transit
- GDPR compliance features
- Regular security vulnerability scanning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards
- Follow SAP CAP best practices
- Use ESLint and Prettier for code formatting
- Write unit tests for new features
- Update documentation for API changes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the [SAP CAP documentation](https://cap.cloud.sap/)
- Review [SAP UI5 documentation](https://ui5.sap.com/)

## 🎯 Roadmap

### Future Enhancements
- AI-powered agent recommendations
- Advanced analytics and reporting
- Mobile application support
- Multi-tenant architecture
- Advanced workflow templates
- Integration with external AI services

---

**Built with ❤️ using SAP BTP, CAP, and UI5**
