import * as vscode from 'vscode';
import { A2AAgentProvider } from './providers/AgentProvider';
import { A2ANetworkProvider } from './providers/NetworkProvider';
import { A2ABlockchainProvider } from './providers/BlockchainProvider';
import { A2AServiceProvider } from './providers/ServiceProvider';
import { A2ADebugDashboard } from './dashboard/DebugDashboard';
import { A2AConfigValidator } from './validators/ConfigValidator';
import { A2ACodeGenerator } from './generators/CodeGenerator';
import { A2ADeploymentManager } from './deployment/DeploymentManager';

export function activate(context: vscode.ExtensionContext) {
    console.log('A2A Framework extension is now active!');

    // Initialize providers
    const agentProvider = new A2AAgentProvider(context);
    const networkProvider = new A2ANetworkProvider(context);
    const blockchainProvider = new A2ABlockchainProvider(context);
    const serviceProvider = new A2AServiceProvider(context);
    
    // Initialize utilities
    const debugDashboard = new A2ADebugDashboard(context);
    const configValidator = new A2AConfigValidator(context);
    const codeGenerator = new A2ACodeGenerator(context);
    const deploymentManager = new A2ADeploymentManager(context);

    // Register tree data providers
    vscode.window.createTreeView('a2a.agentExplorer', {
        treeDataProvider: agentProvider,
        showCollapseAll: true
    });

    vscode.window.createTreeView('a2a.networkView', {
        treeDataProvider: networkProvider,
        showCollapseAll: true
    });

    vscode.window.createTreeView('a2a.blockchainView', {
        treeDataProvider: blockchainProvider,
        showCollapseAll: true
    });

    vscode.window.createTreeView('a2a.servicesView', {
        treeDataProvider: serviceProvider,
        showCollapseAll: true
    });

    // Register commands
    const commands = [
        vscode.commands.registerCommand('a2a.createAgent', () => {
            codeGenerator.createAgent();
        }),

        vscode.commands.registerCommand('a2a.createWorkflow', () => {
            codeGenerator.createWorkflow();
        }),

        vscode.commands.registerCommand('a2a.deployAgent', (uri?: vscode.Uri) => {
            deploymentManager.deployAgent(uri);
        }),

        vscode.commands.registerCommand('a2a.startDebugDashboard', () => {
            debugDashboard.start();
        }),

        vscode.commands.registerCommand('a2a.validateConfig', () => {
            configValidator.validateWorkspace();
        }),

        vscode.commands.registerCommand('a2a.generateService', (uri?: vscode.Uri) => {
            codeGenerator.generateService(uri);
        }),

        vscode.commands.registerCommand('a2a.viewAgentNetwork', () => {
            debugDashboard.showNetworkView();
        }),

        // Refresh commands
        vscode.commands.registerCommand('a2a.refreshAgents', () => {
            agentProvider.refresh();
        }),

        vscode.commands.registerCommand('a2a.refreshNetwork', () => {
            networkProvider.refresh();
        }),

        vscode.commands.registerCommand('a2a.refreshBlockchain', () => {
            blockchainProvider.refresh();
        })
    ];

    // Register all commands
    commands.forEach(command => context.subscriptions.push(command));

    // Auto-validation on save
    const autoValidate = vscode.workspace.onDidSaveTextDocument((document) => {
        if (document.fileName.includes('a2a.config') || 
            document.fileName.includes('.env') ||
            document.fileName.includes('package.json')) {
            configValidator.validateDocument(document);
        }
    });

    context.subscriptions.push(autoValidate);

    // Set context variables
    updateContextVariables();
    
    // Watch for workspace changes
    const workspaceWatcher = vscode.workspace.onDidChangeWorkspaceFolders(() => {
        updateContextVariables();
    });

    context.subscriptions.push(workspaceWatcher);

    // Status bar items
    const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Left, 
        100
    );
    statusBarItem.text = "$(circuit-board) A2A Ready";
    statusBarItem.tooltip = "A2A Framework is active";
    statusBarItem.command = 'a2a.startDebugDashboard';
    statusBarItem.show();

    context.subscriptions.push(statusBarItem);

    // Initialize workspace if A2A project detected
    if (isA2AProject()) {
        vscode.window.showInformationMessage(
            'A2A project detected! Extension features are now available.',
            'Open Dashboard',
            'Validate Config'
        ).then(selection => {
            if (selection === 'Open Dashboard') {
                debugDashboard.start();
            } else if (selection === 'Validate Config') {
                configValidator.validateWorkspace();
            }
        });
    }
}

function updateContextVariables() {
    const isA2A = isA2AProject();
    const hasBlockchain = hasBlockchainConfig();
    
    vscode.commands.executeCommand('setContext', 'a2a.isA2AProject', isA2A);
    vscode.commands.executeCommand('setContext', 'a2a.hasBlockchain', hasBlockchain);
}

function isA2AProject(): boolean {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) return false;

    return workspaceFolders.some(folder => {
        const configFiles = [
            vscode.Uri.joinPath(folder.uri, 'a2a.config.js'),
            vscode.Uri.joinPath(folder.uri, 'a2a.config.json'),
            vscode.Uri.joinPath(folder.uri, 'package.json')
        ];

        return configFiles.some(file => {
            try {
                vscode.workspace.fs.stat(file);
                return true;
            } catch {
                return false;
            }
        });
    });
}

function hasBlockchainConfig(): boolean {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) return false;

    // Check for blockchain configuration in .env or config files
    return workspaceFolders.some(folder => {
        const envFile = vscode.Uri.joinPath(folder.uri, '.env');
        try {
            vscode.workspace.fs.readFile(envFile).then(content => {
                const envContent = Buffer.from(content).toString();
                return envContent.includes('BLOCKCHAIN_ENABLED=true');
            });
            return false;
        } catch {
            return false;
        }
    });
}

export function deactivate() {
    console.log('A2A Framework extension is now deactivated');
}