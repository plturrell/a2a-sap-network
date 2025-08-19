# A2A Network - Rapid Development Setup

🚀 **Get up and running in under 2 minutes!**

## 🎯 Quick Start (Fastest)

### Option 1: Automated Setup (Recommended)

```bash
# Clone and setup everything automatically
git clone <repository-url>
cd a2aNetwork
npm run setup:dev
npm run dev
```

**That's it!** Your development environment is ready at http://localhost:4004

### Option 2: Docker Setup (Zero Dependencies)

```bash
# Start everything with Docker
docker-compose -f docker-compose.dev.yml up
```

Access your services:
- 🖥️ **A2A Network UI**: http://localhost:4004
- 🔗 **Blockchain Explorer**: http://localhost:8081
- 💾 **Database Viewer**: http://localhost:8080
- 🔴 **Redis**: localhost:6379

## 📋 Prerequisites

- **Node.js 18+** (check with `node --version`)
- **npm 9+** (check with `npm --version`)
- **Docker** (optional, for containerized setup)

## 🛠️ Manual Setup (If Needed)

### 1. Install Dependencies
```bash
npm install
```

### 2. Generate Environment
```bash
node scripts/setup-dev-environment.js
```

### 3. Start Development
```bash
npm run dev
```

## 🔧 Development Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start all services for development |
| `npm run setup:dev` | Complete environment setup |
| `npm run start:clean` | **Clean start with port conflict resolution** |
| `npm run dev:clean` | Clean reinstall everything |
| `npm run test:quick` | Run tests quickly |
| `npm run validate` | Lint and test |
| `npm run diagnostic` | System health check |

## 🌟 Features Enabled

✅ **Authentication Bypass** - No login required in development  
✅ **Hot Reload** - Changes reflect instantly  
✅ **Local Blockchain** - Ganache with test contracts  
✅ **SQLite Database** - No external database needed  
✅ **Redis Cache** - Optional, falls back to memory  
✅ **Debug Logging** - Detailed development logs  
✅ **Test Data** - Pre-populated with sample agents  

## 🖼️ User Interface

### Main Launchpad
- **Production**: http://localhost:4004/app/launchpad.html
- **Development**: http://localhost:4004/app/test-launchpad.html

### API Endpoints
- **Health Check**: http://localhost:4004/health
- **API Explorer**: http://localhost:4004/api/v1/network/$metadata
- **Agents API**: http://localhost:4004/api/v1/Agents
- **Services API**: http://localhost:4004/api/v1/Services

## 🔍 Troubleshooting

### Port Already in Use
```bash
# Use the automatic port conflict resolution (RECOMMENDED)
npm run start:clean

# Or manually kill processes on required ports
npx kill-port 4004 8545 6379
npm run dev
```

### Permission Errors
```bash
# Fix script permissions
chmod +x scripts/*.sh scripts/*.js
```

### Environment Issues
```bash
# Reset environment completely
npm run dev:clean
```

### Docker Issues
```bash
# Reset Docker environment
docker-compose -f docker-compose.dev.yml down -v
docker-compose -f docker-compose.dev.yml up --build
```

## 📊 System Health

Run diagnostic check anytime:
```bash
npm run diagnostic
```

This checks:
- ✅ Server connectivity
- ✅ API endpoints
- ✅ File system
- ✅ Database connection
- ✅ Blockchain status

## 🔐 Authentication

### Development Mode
- **Enabled by default** - No authentication required
- **Test user**: `dev-user` with full permissions
- **All APIs accessible** without tokens

### Production Mode
- XSUAA authentication required
- JWT tokens validated
- Role-based access control

## 📁 Project Structure

```
a2aNetwork/
├── app/                    # Frontend applications
│   ├── launchpad.html     # Production launchpad
│   ├── test-launchpad.html # Development launchpad
│   └── a2aFiori/          # Main Fiori app
├── srv/                   # Backend services
│   ├── server.js          # Main server
│   ├── middleware/        # Security, auth, etc.
│   └── services/          # Business logic
├── db/                    # Database schemas
├── scripts/               # Automation scripts
│   ├── setup-dev-environment.js
│   └── quick-start.js
├── data/                  # Development data
├── docker-compose.dev.yml # Docker development setup
└── .env                   # Environment variables
```

## 🚀 Next Steps

After setup, you can:

1. **Explore the UI** at http://localhost:4004
2. **Test API endpoints** using the browser or Postman
3. **View blockchain transactions** at http://localhost:8081
4. **Check database content** at http://localhost:8080
5. **Review system logs** in the terminal

## 🆘 Getting Help

- **Health Check**: `npm run diagnostic`
- **View Logs**: Check terminal output or `logs/` directory
- **Reset Everything**: `npm run dev:clean`
- **Check Status**: `curl http://localhost:4004/health`

## 🎯 Production Deployment

For production deployment to SAP BTP:

```bash
# Build for production
npm run build

# Deploy to Cloud Foundry
npm run deploy:cf

# Deploy MTA
mbt build && cf deploy mta_archives/a2a-network_1.0.0.mtar
```

---

**Happy coding! 🚀**

*This setup is optimized for rapid development. All security features are properly configured for production deployment.*