const k8s = require('@kubernetes/client-node');
const yaml = require('js-yaml');
const fs = require('fs').promises;
const path = require('path');
const chalk = require('chalk');
const ora = require('ora');
const { v4: uuidv4 } = require('uuid');

class KubernetesDeployer {
  constructor() {
    this.kc = new k8s.KubeConfig();
    this.loadConfig();
    this.k8sApi = this.kc.makeApiClient(k8s.AppsV1Api);
    this.coreApi = this.kc.makeApiClient(k8s.CoreV1Api);
    this.customApi = this.kc.makeApiClient(k8s.CustomObjectsApi);
  }

  loadConfig() {
    try {
      this.kc.loadFromDefault();
    } catch (error) {
      try {
        this.kc.loadFromCluster();
      } catch (clusterError) {
        throw new Error('Could not load Kubernetes configuration. Make sure kubectl is configured.');
      }
    }
  }

  async deploy(options) {
    const spinner = ora('Preparing deployment...').start();
    
    try {
      // Load deployment configuration
      const config = await this.loadDeploymentConfig(options.config);
      
      // Validate configuration
      await this.validateConfig(config);
      
      spinner.text = 'Creating namespace...';
      await this.ensureNamespace(config.namespace || 'a2a-system');
      
      spinner.text = 'Deploying agents...';
      const deploymentResults = [];
      
      for (const agent of config.agents) {
        const result = await this.deployAgent(agent, config);
        deploymentResults.push(result);
      }
      
      spinner.text = 'Creating services...';
      const serviceResults = [];
      
      for (const service of config.services || []) {
        const result = await this.createService(service, config);
        serviceResults.push(result);
      }
      
      spinner.text = 'Setting up ingress...';
      const ingressResults = [];
      
      if (config.ingress) {
        const result = await this.createIngress(config.ingress, config);
        ingressResults.push(result);
      }
      
      spinner.text = 'Applying configurations...';
      const configResults = [];
      
      if (config.configMaps) {
        for (const configMap of config.configMaps) {
          const result = await this.createConfigMap(configMap, config);
          configResults.push(result);
        }
      }
      
      if (config.secrets) {
        for (const secret of config.secrets) {
          const result = await this.createSecret(secret, config);
          configResults.push(result);
        }
      }
      
      spinner.succeed('Deployment completed successfully!');
      
      return {
        deploymentId: uuidv4(),
        namespace: config.namespace || 'a2a-system',
        agents: deploymentResults,
        services: serviceResults,
        ingress: ingressResults,
        configs: configResults,
        timestamp: new Date().toISOString()
      };
      
    } catch (error) {
      spinner.fail('Deployment failed');
      throw error;
    }
  }

  async deployAgent(agentConfig, globalConfig) {
    const namespace = globalConfig.namespace || 'a2a-system';
    
    // Generate Kubernetes deployment manifest
    const deployment = this.generateDeployment(agentConfig, globalConfig);
    
    try {
      // Try to update existing deployment
      await this.k8sApi.patchNamespacedDeployment(
        agentConfig.name,
        namespace,
        deployment,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        { headers: { 'Content-Type': 'application/strategic-merge-patch+json' } }
      );
      
      return {
        name: agentConfig.name,
        action: 'updated',
        status: 'success'
      };
      
    } catch (error) {
      if (error.response?.statusCode === 404) {
        // Create new deployment
        await this.k8sApi.createNamespacedDeployment(namespace, deployment);
        
        return {
          name: agentConfig.name,
          action: 'created',
          status: 'success'
        };
      } else {
        throw error;
      }
    }
  }

  generateDeployment(agentConfig, globalConfig) {
    const namespace = globalConfig.namespace || 'a2a-system';
    
    return {
      apiVersion: 'apps/v1',
      kind: 'Deployment',
      metadata: {
        name: agentConfig.name,
        namespace: namespace,
        labels: {
          app: agentConfig.name,
          'app.kubernetes.io/name': agentConfig.name,
          'app.kubernetes.io/component': 'a2a-agent',
          'app.kubernetes.io/managed-by': 'a2a-deploy',
          'a2a.io/agent-type': agentConfig.type || 'generic'
        }
      },
      spec: {
        replicas: agentConfig.replicas || 1,
        selector: {
          matchLabels: {
            app: agentConfig.name
          }
        },
        template: {
          metadata: {
            labels: {
              app: agentConfig.name,
              'app.kubernetes.io/name': agentConfig.name,
              'app.kubernetes.io/component': 'a2a-agent'
            }
          },
          spec: {
            containers: [
              {
                name: agentConfig.name,
                image: agentConfig.image || `a2a/${agentConfig.name}:latest`,
                ports: [
                  {
                    containerPort: agentConfig.port || 3000,
                    name: 'http'
                  }
                ],
                env: this.generateEnvironmentVariables(agentConfig, globalConfig),
                resources: agentConfig.resources || {
                  requests: {
                    memory: '128Mi',
                    cpu: '100m'
                  },
                  limits: {
                    memory: '512Mi',
                    cpu: '500m'
                  }
                },
                livenessProbe: {
                  httpGet: {
                    path: '/health',
                    port: 'http'
                  },
                  initialDelaySeconds: 30,
                  periodSeconds: 10
                },
                readinessProbe: {
                  httpGet: {
                    path: '/ready',
                    port: 'http'
                  },
                  initialDelaySeconds: 5,
                  periodSeconds: 5
                },
                volumeMounts: agentConfig.volumeMounts || []
              }
            ],
            volumes: agentConfig.volumes || [],
            restartPolicy: 'Always',
            serviceAccountName: agentConfig.serviceAccount || 'a2a-agent'
          }
        }
      }
    };
  }

  generateEnvironmentVariables(agentConfig, globalConfig) {
    const baseEnv = [
      {
        name: 'NODE_ENV',
        value: globalConfig.environment || 'production'
      },
      {
        name: 'AGENT_NAME',
        value: agentConfig.name
      },
      {
        name: 'AGENT_TYPE',
        value: agentConfig.type || 'generic'
      },
      {
        name: 'A2A_REGISTRY_URL',
        value: globalConfig.registryUrl || 'http://a2a-registry:3000'
      },
      {
        name: 'POD_NAME',
        valueFrom: {
          fieldRef: {
            fieldPath: 'metadata.name'
          }
        }
      },
      {
        name: 'POD_NAMESPACE',
        valueFrom: {
          fieldRef: {
            fieldPath: 'metadata.namespace'
          }
        }
      }
    ];

    // Add custom environment variables
    if (agentConfig.env) {
      for (const [key, value] of Object.entries(agentConfig.env)) {
        baseEnv.push({
          name: key,
          value: value.toString()
        });
      }
    }

    // Add secret references
    if (agentConfig.secretRefs) {
      for (const secretRef of agentConfig.secretRefs) {
        baseEnv.push({
          name: secretRef.name,
          valueFrom: {
            secretKeyRef: {
              name: secretRef.secret,
              key: secretRef.key
            }
          }
        });
      }
    }

    return baseEnv;
  }

  async createService(serviceConfig, globalConfig) {
    const namespace = globalConfig.namespace || 'a2a-system';
    
    const service = {
      apiVersion: 'v1',
      kind: 'Service',
      metadata: {
        name: serviceConfig.name,
        namespace: namespace,
        labels: {
          app: serviceConfig.name,
          'app.kubernetes.io/component': 'a2a-service'
        }
      },
      spec: {
        selector: {
          app: serviceConfig.selector || serviceConfig.name
        },
        ports: serviceConfig.ports || [
          {
            port: 80,
            targetPort: 'http',
            protocol: 'TCP',
            name: 'http'
          }
        ],
        type: serviceConfig.type || 'ClusterIP'
      }
    };

    try {
      await this.coreApi.createNamespacedService(namespace, service);
      return {
        name: serviceConfig.name,
        action: 'created',
        status: 'success'
      };
    } catch (error) {
      if (error.response?.statusCode === 409) {
        // Service already exists, update it
        await this.coreApi.patchNamespacedService(
          serviceConfig.name,
          namespace,
          service,
          undefined,
          undefined,
          undefined,
          undefined,
          undefined,
          { headers: { 'Content-Type': 'application/strategic-merge-patch+json' } }
        );
        
        return {
          name: serviceConfig.name,
          action: 'updated',
          status: 'success'
        };
      } else {
        throw error;
      }
    }
  }

  async createIngress(ingressConfig, globalConfig) {
    const namespace = globalConfig.namespace || 'a2a-system';
    
    const ingress = {
      apiVersion: 'networking.k8s.io/v1',
      kind: 'Ingress',
      metadata: {
        name: ingressConfig.name || 'a2a-ingress',
        namespace: namespace,
        annotations: ingressConfig.annotations || {
          'nginx.ingress.kubernetes.io/rewrite-target': '/',
          'cert-manager.io/cluster-issuer': 'letsencrypt-prod'
        }
      },
      spec: {
        tls: ingressConfig.tls || [],
        rules: ingressConfig.rules || []
      }
    };

    try {
      const networkingApi = this.kc.makeApiClient(k8s.NetworkingV1Api);
      await networkingApi.createNamespacedIngress(namespace, ingress);
      
      return {
        name: ingressConfig.name || 'a2a-ingress',
        action: 'created',
        status: 'success'
      };
    } catch (error) {
      if (error.response?.statusCode === 409) {
        const networkingApi = this.kc.makeApiClient(k8s.NetworkingV1Api);
        await networkingApi.patchNamespacedIngress(
          ingressConfig.name || 'a2a-ingress',
          namespace,
          ingress
        );
        
        return {
          name: ingressConfig.name || 'a2a-ingress',
          action: 'updated',
          status: 'success'
        };
      } else {
        throw error;
      }
    }
  }

  async createConfigMap(configMapConfig, globalConfig) {
    const namespace = globalConfig.namespace || 'a2a-system';
    
    const configMap = {
      apiVersion: 'v1',
      kind: 'ConfigMap',
      metadata: {
        name: configMapConfig.name,
        namespace: namespace
      },
      data: configMapConfig.data || {}
    };

    try {
      await this.coreApi.createNamespacedConfigMap(namespace, configMap);
      return {
        name: configMapConfig.name,
        action: 'created',
        status: 'success'
      };
    } catch (error) {
      if (error.response?.statusCode === 409) {
        await this.coreApi.patchNamespacedConfigMap(
          configMapConfig.name,
          namespace,
          configMap
        );
        
        return {
          name: configMapConfig.name,
          action: 'updated',
          status: 'success'
        };
      } else {
        throw error;
      }
    }
  }

  async createSecret(secretConfig, globalConfig) {
    const namespace = globalConfig.namespace || 'a2a-system';
    
    // Encode secret data in base64
    const encodedData = {};
    for (const [key, value] of Object.entries(secretConfig.data || {})) {
      encodedData[key] = Buffer.from(value.toString()).toString('base64');
    }
    
    const secret = {
      apiVersion: 'v1',
      kind: 'Secret',
      metadata: {
        name: secretConfig.name,
        namespace: namespace
      },
      type: secretConfig.type || 'Opaque',
      data: encodedData
    };

    try {
      await this.coreApi.createNamespacedSecret(namespace, secret);
      return {
        name: secretConfig.name,
        action: 'created',
        status: 'success'
      };
    } catch (error) {
      if (error.response?.statusCode === 409) {
        await this.coreApi.patchNamespacedSecret(
          secretConfig.name,
          namespace,
          secret
        );
        
        return {
          name: secretConfig.name,
          action: 'updated',
          status: 'success'
        };
      } else {
        throw error;
      }
    }
  }

  async ensureNamespace(namespace) {
    const namespaceObj = {
      apiVersion: 'v1',
      kind: 'Namespace',
      metadata: {
        name: namespace,
        labels: {
          'app.kubernetes.io/managed-by': 'a2a-deploy'
        }
      }
    };

    try {
      await this.coreApi.createNamespace(namespaceObj);
    } catch (error) {
      if (error.response?.statusCode !== 409) {
        throw error;
      }
      // Namespace already exists, continue
    }
  }

  async loadDeploymentConfig(configFile) {
    try {
      const configContent = await fs.readFile(configFile, 'utf8');
      
      if (configFile.endsWith('.yaml') || configFile.endsWith('.yml')) {
        return yaml.load(configContent);
      } else if (configFile.endsWith('.json')) {
        return JSON.parse(configContent);
      } else {
        throw new Error('Unsupported config file format. Use .yaml, .yml, or .json');
      }
    } catch (error) {
      throw new Error(`Failed to load deployment config: ${error.message}`);
    }
  }

  async validateConfig(config) {
    if (!config.agents || !Array.isArray(config.agents)) {
      throw new Error('Deployment config must contain an "agents" array');
    }

    for (const agent of config.agents) {
      if (!agent.name) {
        throw new Error('Each agent must have a "name" field');
      }
      
      if (!agent.image && !agent.dockerfile) {
        throw new Error(`Agent "${agent.name}" must specify either "image" or "dockerfile"`);
      }
    }
  }

  async rollback(options) {
    const namespace = options.environment || 'a2a-system';
    const deploymentName = options.deployment;
    
    if (!deploymentName) {
      throw new Error('Deployment name is required for rollback');
    }

    const spinner = ora(`Rolling back deployment ${deploymentName}...`).start();

    try {
      // Get rollout history
      const rolloutHistory = await this.k8sApi.listNamespacedReplicaSet(
        namespace,
        undefined,
        undefined,
        undefined,
        undefined,
        `app=${deploymentName}`
      );

      if (rolloutHistory.body.items.length < 2) {
        throw new Error('No previous version available for rollback');
      }

      // Perform rollback
      const rollbackBody = {
        spec: {
          rollbackTo: {
            revision: options.version ? parseInt(options.version) : 0
          }
        }
      };

      await this.k8sApi.patchNamespacedDeployment(
        deploymentName,
        namespace,
        rollbackBody,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        { headers: { 'Content-Type': 'application/strategic-merge-patch+json' } }
      );

      spinner.succeed(`Rollback completed for deployment ${deploymentName}`);

      return {
        deployment: deploymentName,
        namespace: namespace,
        action: 'rollback',
        status: 'success',
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      spinner.fail('Rollback failed');
      throw error;
    }
  }

  async scale(options) {
    const namespace = options.environment || 'a2a-system';
    const deploymentName = options.deployment;
    const replicas = parseInt(options.replicas);

    if (!deploymentName) {
      throw new Error('Deployment name is required for scaling');
    }

    const spinner = ora(`Scaling deployment ${deploymentName} to ${replicas} replicas...`).start();

    try {
      const scaleBody = {
        spec: {
          replicas: replicas
        }
      };

      await this.k8sApi.patchNamespacedDeploymentScale(
        deploymentName,
        namespace,
        scaleBody
      );

      spinner.succeed(`Scaled deployment ${deploymentName} to ${replicas} replicas`);

      return {
        deployment: deploymentName,
        namespace: namespace,
        replicas: replicas,
        action: 'scale',
        status: 'success',
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      spinner.fail('Scaling failed');
      throw error;
    }
  }
}

module.exports = { KubernetesDeployer };