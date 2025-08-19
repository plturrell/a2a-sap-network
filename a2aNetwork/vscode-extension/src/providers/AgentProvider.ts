import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

export class A2AAgentProvider implements vscode.TreeDataProvider<AgentItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<AgentItem | undefined | null | void> = new vscode.EventEmitter<AgentItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<AgentItem | undefined | null | void> = this._onDidChangeTreeData.event;

    constructor(private context: vscode.ExtensionContext) {}

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: AgentItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: AgentItem): Thenable<AgentItem[]> {
        if (!element) {
            return Promise.resolve(this.getAgents());
        } else {
            return Promise.resolve(this.getAgentDetails(element));
        }
    }

    private async getAgents(): Promise<AgentItem[]> {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            return [];
        }

        const agents: AgentItem[] = [];

        // Look for agents in various locations
        const agentPaths = [
            'src/index.js',
            'agents',
            'src/agents'
        ];

        for (const agentPath of agentPaths) {
            const fullPath = path.join(workspaceFolder.uri.fsPath, agentPath);
            
            try {
                if (fs.existsSync(fullPath)) {
                    if (fs.statSync(fullPath).isDirectory()) {
                        // Scan directory for agents
                        const agentDirs = fs.readdirSync(fullPath);
                        for (const dir of agentDirs) {
                            const agentDir = path.join(fullPath, dir);
                            if (fs.statSync(agentDir).isDirectory()) {
                                const agent = await this.parseAgentDirectory(agentDir, dir);
                                if (agent) agents.push(agent);
                            }
                        }
                    } else {
                        // Single agent file
                        const agent = await this.parseAgentFile(fullPath);
                        if (agent) agents.push(agent);
                    }
                }
            } catch (error) {
                console.error(`Error scanning ${fullPath}:`, error);
            }
        }

        // Add running agents from registry
        const runningAgents = await this.getRunningAgents();
        agents.push(...runningAgents);

        return agents;
    }

    private async parseAgentDirectory(dirPath: string, name: string): Promise<AgentItem | null> {
        const indexFile = path.join(dirPath, 'index.js');
        const packageFile = path.join(dirPath, 'package.json');
        
        let agentInfo: any = { name, type: 'unknown' };

        if (fs.existsSync(packageFile)) {
            try {
                const packageContent = fs.readFileSync(packageFile, 'utf8');
                const packageJson = JSON.parse(packageContent);
                agentInfo = { ...agentInfo, ...packageJson };
            } catch (error) {
                console.error(`Error parsing ${packageFile}:`, error);
            }
        }

        if (fs.existsSync(indexFile)) {
            const agentCode = fs.readFileSync(indexFile, 'utf8');
            agentInfo = { ...agentInfo, ...this.parseAgentCode(agentCode) };
        }

        return new AgentItem(
            agentInfo.name || name,
            agentInfo.type || 'unknown',
            'local',
            vscode.TreeItemCollapsibleState.Collapsed,
            {
                capabilities: agentInfo.capabilities || [],
                path: dirPath,
                status: 'stopped'
            }
        );
    }

    private async parseAgentFile(filePath: string): Promise<AgentItem | null> {
        try {
            const agentCode = fs.readFileSync(filePath, 'utf8');
            const agentInfo = this.parseAgentCode(agentCode);
            
            return new AgentItem(
                agentInfo.name || 'Main Agent',
                agentInfo.type || 'unknown',
                'local',
                vscode.TreeItemCollapsibleState.Collapsed,
                {
                    capabilities: agentInfo.capabilities || [],
                    path: filePath,
                    status: 'stopped'
                }
            );
        } catch (error) {
            console.error(`Error parsing ${filePath}:`, error);
            return null;
        }
    }

    private parseAgentCode(code: string): any {
        const info: any = {};
        
        // Extract agent name
        const nameMatch = code.match(/name:\s*['"`]([^'"`]+)['"`]/);
        if (nameMatch) info.name = nameMatch[1];

        // Extract agent type
        const typeMatch = code.match(/type:\s*['"`]([^'"`]+)['"`]/);
        if (typeMatch) info.type = typeMatch[1];

        // Extract capabilities
        const capabilitiesMatch = code.match(/capabilities:\s*\[(.*?)\]/s);
        if (capabilitiesMatch) {
            try {
                const capArray = capabilitiesMatch[1]
                    .split(',')
                    .map(cap => cap.trim().replace(/['"`]/g, ''))
                    .filter(cap => cap.length > 0);
                info.capabilities = capArray;
            } catch (error) {
                info.capabilities = [];
            }
        }

        return info;
    }

    private async getRunningAgents(): Promise<AgentItem[]> {
        // Connect to local registry and get running agents
        try {
            const config = vscode.workspace.getConfiguration('a2a');
            const registryUrl = config.get('registryUrl', 'http://localhost:3000');
            
            // This would typically make an HTTP request to the registry
            // For now, return empty array
            return [];
        } catch (error) {
            return [];
        }
    }

    private getAgentDetails(agent: AgentItem): AgentItem[] {
        const details: AgentItem[] = [];

        if (agent.metadata?.capabilities?.length) {
            details.push(new AgentItem(
                'Capabilities',
                'capabilities',
                'info',
                vscode.TreeItemCollapsibleState.Collapsed,
                { capabilities: agent.metadata.capabilities }
            ));
        }

        details.push(new AgentItem(
            `Status: ${agent.metadata?.status || 'unknown'}`,
            'status',
            'info',
            vscode.TreeItemCollapsibleState.None
        ));

        if (agent.metadata?.path) {
            details.push(new AgentItem(
                'Open File',
                'file',
                'action',
                vscode.TreeItemCollapsibleState.None,
                { path: agent.metadata.path }
            ));
        }

        return details;
    }
}

export class AgentItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly type: string,
        public readonly category: 'local' | 'remote' | 'info' | 'action',
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly metadata?: any
    ) {
        super(label, collapsibleState);

        this.tooltip = this.getTooltip();
        this.description = this.getDescription();
        this.iconPath = this.getIcon();
        this.contextValue = this.getContextValue();
        
        if (category === 'action' && metadata?.path) {
            this.command = {
                command: 'vscode.open',
                title: 'Open',
                arguments: [vscode.Uri.file(metadata.path)]
            };
        }
    }

    private getTooltip(): string {
        switch (this.category) {
            case 'local':
                return `Local agent: ${this.label} (${this.type})`;
            case 'remote':
                return `Remote agent: ${this.label}`;
            case 'info':
                return this.label;
            case 'action':
                return `Click to ${this.label.toLowerCase()}`;
            default:
                return this.label;
        }
    }

    private getDescription(): string {
        if (this.category === 'local' || this.category === 'remote') {
            return this.type;
        }
        return '';
    }

    private getIcon(): vscode.ThemeIcon {
        switch (this.category) {
            case 'local':
                return new vscode.ThemeIcon('robot');
            case 'remote':
                return new vscode.ThemeIcon('cloud');
            case 'info':
                return new vscode.ThemeIcon('info');
            case 'action':
                return new vscode.ThemeIcon('file-code');
            default:
                return new vscode.ThemeIcon('circle-outline');
        }
    }

    private getContextValue(): string {
        return `agent.${this.category}.${this.type}`;
    }
}