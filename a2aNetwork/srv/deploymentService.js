const cds = require('@sap/cds');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;

module.exports = class DeploymentService extends cds.ApplicationService {
    
    async init() {
        const { Deployments, DeploymentStages, HealthChecks, Notifications } = this.entities;
        
        // Add virtual fields
        this.after('READ', 'Deployments', deployments => {
            const addVirtualFields = deployment => {
                if (!deployment) return;
                
                // Status icon and color
                switch (deployment.status) {
                    case 'completed':
                        deployment.statusIcon = 'sap-icon://accept';
                        deployment.statusColor = 'Success';
                        break;
                    case 'failed':
                        deployment.statusIcon = 'sap-icon://error';
                        deployment.statusColor = 'Error';
                        break;
                    case 'in_progress':
                        deployment.statusIcon = 'sap-icon://in-progress';
                        deployment.statusColor = 'Warning';
                        break;
                    case 'rolled_back':
                        deployment.statusIcon = 'sap-icon://undo';
                        deployment.statusColor = 'Critical';
                        break;
                    default:
                        deployment.statusIcon = 'sap-icon://pending';
                        deployment.statusColor = 'None';
                }
                
                // Progress percentage
                if (deployment.status === 'completed') {
                    deployment.progressPercentage = 100;
                } else if (deployment.status === 'in_progress' && deployment.stages) {
                    const completed = deployment.stages.filter(s => s.status === 'completed').length;
                    const total = deployment.stages.length;
                    deployment.progressPercentage = total > 0 ? Math.round((completed / total) * 100) : 0;
                } else {
                    deployment.progressPercentage = 0;
                }
            };
            
            if (Array.isArray(deployments)) {
                deployments.forEach(addVirtualFields);
            } else {
                addVirtualFields(deployments);
            }
        });
        
        // Action: Start deployment
        this.on('start', 'Deployments', async (req) => {
            const { ID } = req.params[0];
            const deployment = await SELECT.one.from(Deployments).where({ ID });
            
            if (!deployment) {
                req.error(404, 'Deployment not found');
            }
            
            if (deployment.status !== 'pending') {
                req.error(400, 'Deployment must be in pending status to start');
            }
            
            // Update deployment status
            await UPDATE(Deployments, ID).with({
                status: 'in_progress',
                startTime: new Date()
            });
            
            // Emit event
            await this.emit('deploymentStarted', {
                deploymentId: ID,
                appName: deployment.appName,
                environment: deployment.environment,
                version: deployment.version,
                initiatedBy: req.user.id
            });
            
            // Start async deployment process
            this._executeDeployment(ID).catch(console.error);
            
            return SELECT.one.from(Deployments).where({ ID });
        });
        
        // Action: Approve deployment
        this.on('approve', 'Deployments', async (req) => {
            const { ID } = req.params[0];
            
            await UPDATE(Deployments, ID).with({
                approvedBy: req.user.id,
                notes: `Approved by ${req.user.id} at ${new Date().toISOString()}`
            });
            
            return SELECT.one.from(Deployments).where({ ID });
        });
        
        // Action: Rollback deployment
        this.on('rollback', 'Deployments', async (req) => {
            const { ID } = req.params[0];
            const deployment = await SELECT.one.from(Deployments).where({ ID });
            
            if (!deployment) {
                req.error(404, 'Deployment not found');
            }
            
            // Find previous successful deployment
            const previousDeployment = await SELECT.one.from(Deployments)
                .where({
                    appName: deployment.appName,
                    environment: deployment.environment,
                    status: 'completed',
                    ID: { '!=': ID }
                })
                .orderBy('endTime desc');
            
            if (!previousDeployment) {
                req.error(404, 'No previous successful deployment found');
            }
            
            // Create rollback record
            await INSERT.into('com.sap.a2a.deployment.Rollbacks').entries({
                deployment_ID: ID,
                rollbackFrom_ID: ID,
                rollbackTo_ID: previousDeployment.ID,
                reason: req.data.reason || 'Manual rollback',
                initiatedBy: req.user.id,
                timestamp: new Date(),
                status: 'initiated'
            });
            
            // Start rollback process
            this._executeRollback(ID, previousDeployment.ID).catch(console.error);
            
            return SELECT.one.from(Deployments).where({ ID });
        });
        
        // Function: Get deployment status
        this.on('getDeploymentStatus', async (req) => {
            const { environment } = req.data;
            
            const lastDeployment = await SELECT.one.from(Deployments)
                .where({ environment })
                .orderBy('startTime desc');
            
            const activeDeployments = await SELECT.from(Deployments)
                .where({ environment, status: 'in_progress' })
                .count();
            
            // Calculate health and uptime
            const healthChecks = lastDeployment ? 
                await SELECT.from(HealthChecks).where({ deployment_ID: lastDeployment.ID }) : [];
            
            const healthyChecks = healthChecks.filter(hc => hc.status === 'healthy').length;
            const healthStatus = healthChecks.length > 0 ? 
                (healthyChecks === healthChecks.length ? 'healthy' : 'degraded') : 'unknown';
            
            const uptime = lastDeployment && lastDeployment.endTime ? 
                ((new Date() - new Date(lastDeployment.endTime)) / (1000 * 60 * 60 * 24)).toFixed(2) : 0;
            
            return {
                environment,
                lastDeployment,
                activeDeployments: activeDeployments.count,
                healthStatus,
                uptime: parseFloat(uptime)
            };
        });
        
        // Function: Get live deployment status
        this.on('getLiveDeploymentStatus', async () => {
            const activeDeployments = await SELECT.from(Deployments)
                .where({ status: 'in_progress' })
                .columns('*', { ref: ['stages'], expand: ['*'] });
            
            return {
                activeDeployments: activeDeployments.map(d => {
                    const currentStage = d.stages?.find(s => s.status === 'running')?.stageName || 'Initializing';
                    const progress = d.stages?.length > 0 ? 
                        Math.round((d.stages.filter(s => s.status === 'completed').length / d.stages.length) * 100) : 0;
                    
                    return {
                        id: d.ID,
                        appName: d.appName,
                        environment: d.environment,
                        status: d.status,
                        progress,
                        currentStage,
                        startTime: d.startTime,
                        estimatedCompletion: this._estimateCompletion(d)
                    };
                })
            };
        });
        
        // Action: Deploy to Fly.io
        this.on('deployToFly', async (req) => {
            const { appName, environment, strategy } = req.data;
            
            // Create deployment record
            const deployment = await INSERT.into(Deployments).entries({
                appName,
                environment,
                version: await this._getLatestVersion(appName),
                deploymentType: strategy || 'standard',
                status: 'pending',
                deployedBy: req.user.id
            });
            
            // Execute deployment script
            const scriptPath = path.join(__dirname, '../../scripts/deployment/fly');
            const scriptName = strategy === 'zero_downtime' ? 
                'deploy-fly-zero-downtime.sh' : 'deploy-fly.sh';
            
            this._runScript(path.join(scriptPath, scriptName), [appName], deployment.ID);
            
            return {
                deploymentId: deployment.ID,
                status: 'started',
                message: `Deployment ${deployment.ID} initiated`
            };
        });
        
        // Function: Get system health
        this.on('getSystemHealth', async () => {
            const healthData = async (env) => {
                // Get agent health from monitoring endpoint
                try {
                    const response = await fetch('http://localhost:8000/api/v1/monitoring/agents/status');
                    const data = await response.json();
                    
                    const healthy = data.agents.filter(a => a.health.status === 'healthy').length;
                    const total = data.agents.length;
                    
                    return {
                        status: healthy === total ? 'healthy' : healthy > total/2 ? 'degraded' : 'critical',
                        healthScore: Math.round((healthy / total) * 100),
                        activeAgents: healthy,
                        totalAgents: total
                    };
                } catch (error) {
                    return {
                        status: 'unknown',
                        healthScore: 0,
                        activeAgents: 0,
                        totalAgents: 18
                    };
                }
            };
            
            return {
                production: await healthData('production'),
                staging: await healthData('staging'),
                alerts: await this._getRecentAlerts()
            };
        });
        
        return super.init();
    }
    
    // Private helper methods
    async _executeDeployment(deploymentId) {
        const { Deployments, DeploymentStages } = this.entities;
        
        try {
            // Define deployment stages
            const stages = [
                'Pre-deployment validation',
                'Build Docker image',
                'Push to registry',
                'Deploy to target',
                'Health checks',
                'Post-deployment tasks'
            ];
            
            // Create stage records
            for (let i = 0; i < stages.length; i++) {
                await INSERT.into(DeploymentStages).entries({
                    deployment_ID: deploymentId,
                    stageName: stages[i],
                    stageOrder: i + 1,
                    status: 'pending'
                });
            }
            
            // Execute each stage
            for (let i = 0; i < stages.length; i++) {
                await this._executeStage(deploymentId, stages[i]);
            }
            
            // Update deployment as completed
            await UPDATE(Deployments, deploymentId).with({
                status: 'completed',
                endTime: new Date(),
                duration: Math.round((new Date() - deployment.startTime) / 1000)
            });
            
            // Emit completion event
            await this.emit('deploymentCompleted', {
                deploymentId,
                appName: deployment.appName,
                environment: deployment.environment,
                status: 'completed',
                duration: deployment.duration
            });
            
        } catch (error) {
            // Handle failure
            await UPDATE(Deployments, deploymentId).with({
                status: 'failed',
                endTime: new Date()
            });
            
            await this.emit('deploymentFailed', {
                deploymentId,
                error: error.message,
                stage: error.stage || 'unknown'
            });
        }
    }
    
    async _executeStage(deploymentId, stageName) {
        const { DeploymentStages } = this.entities;
        
        const stage = await SELECT.one.from(DeploymentStages)
            .where({ deployment_ID: deploymentId, stageName });
        
        try {
            await UPDATE(DeploymentStages, stage.ID).with({
                status: 'running',
                startTime: new Date()
            });
            
            // Simulate stage execution (replace with actual logic)
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            await UPDATE(DeploymentStages, stage.ID).with({
                status: 'completed',
                endTime: new Date()
            });
            
        } catch (error) {
            await UPDATE(DeploymentStages, stage.ID).with({
                status: 'failed',
                endTime: new Date(),
                errorMessage: error.message
            });
            throw error;
        }
    }
    
    _runScript(scriptPath, args, deploymentId) {
        const script = spawn(scriptPath, args);
        
        script.stdout.on('data', (data) => {
            console.log(`[Deployment ${deploymentId}] ${data}`);
        });
        
        script.stderr.on('data', (data) => {
            console.error(`[Deployment ${deploymentId}] Error: ${data}`);
        });
        
        script.on('close', (code) => {
            console.log(`[Deployment ${deploymentId}] Script exited with code ${code}`);
        });
    }
    
    _estimateCompletion(deployment) {
        if (!deployment.startTime || !deployment.stages) return null;
        
        const avgStageDuration = 120; // 2 minutes per stage
        const remainingStages = deployment.stages.filter(s => s.status !== 'completed').length;
        const estimatedSeconds = remainingStages * avgStageDuration;
        
        return new Date(new Date().getTime() + estimatedSeconds * 1000);
    }
    
    async _getLatestVersion(appName) {
        // Get version from git or package.json
        try {
            const packagePath = path.join(__dirname, '../../package.json');
            const packageJson = JSON.parse(await fs.readFile(packagePath, 'utf8'));
            return packageJson.version || '1.0.0';
        } catch {
            return '1.0.0';
        }
    }
    
    async _getRecentAlerts() {
        // Fetch alerts from monitoring system
        try {
            const response = await fetch('http://localhost:8000/api/v1/monitoring/alerts');
            const data = await response.json();
            return data.alerts || [];
        } catch {
            return [];
        }
    }
};