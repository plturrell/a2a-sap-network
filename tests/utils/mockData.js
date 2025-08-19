// A2A Enterprise Mock Data
// Centralized mock data for consistent testing

/**
 * Agent mock data
 */
const agents = {
  reasoning: {
    id: 'agent-reasoning-001',
    name: 'Reasoning Agent',
    type: 'reasoning',
    status: 'active',
    capabilities: ['logical_reasoning', 'problem_solving', 'analysis'],
    configuration: {
      maxTokens: 4000,
      temperature: 0.7,
      modelVersion: '1.0.0'
    },
    metadata: {
      created: '2024-01-01T00:00:00Z',
      updated: '2024-01-01T00:00:00Z',
      version: '1.0.0'
    }
  },
  
  search: {
    id: 'agent-search-001',
    name: 'Search Agent',
    type: 'search',
    status: 'active',
    capabilities: ['web_search', 'document_search', 'knowledge_retrieval'],
    configuration: {
      searchDepth: 10,
      resultLimit: 50,
      enableCache: true
    },
    metadata: {
      created: '2024-01-01T00:00:00Z',
      updated: '2024-01-01T00:00:00Z',
      version: '1.0.0'
    }
  },
  
  inactive: {
    id: 'agent-inactive-001',
    name: 'Inactive Agent',
    type: 'utility',
    status: 'inactive',
    capabilities: ['data_processing'],
    configuration: {},
    metadata: {
      created: '2024-01-01T00:00:00Z',
      updated: '2024-01-01T00:00:00Z',
      version: '1.0.0'
    }
  }
};

/**
 * User mock data
 */
const users = {
  admin: {
    id: 'user-admin-001',
    username: 'admin',
    email: 'admin@example.com',
    role: 'administrator',
    permissions: ['read', 'write', 'delete', 'admin'],
    profile: {
      firstName: 'Admin',
      lastName: 'User',
      department: 'IT'
    },
    metadata: {
      created: '2024-01-01T00:00:00Z',
      lastLogin: '2024-01-01T00:00:00Z'
    }
  },
  
  regular: {
    id: 'user-regular-001',
    username: 'user',
    email: 'user@example.com',
    role: 'user',
    permissions: ['read', 'write'],
    profile: {
      firstName: 'Regular',
      lastName: 'User',
      department: 'Business'
    },
    metadata: {
      created: '2024-01-01T00:00:00Z',
      lastLogin: '2024-01-01T00:00:00Z'
    }
  },
  
  readonly: {
    id: 'user-readonly-001',
    username: 'readonly',
    email: 'readonly@example.com',
    role: 'viewer',
    permissions: ['read'],
    profile: {
      firstName: 'Read',
      lastName: 'Only',
      department: 'Audit'
    },
    metadata: {
      created: '2024-01-01T00:00:00Z',
      lastLogin: '2024-01-01T00:00:00Z'
    }
  }
};

/**
 * Blockchain mock data
 */
const blockchain = {
  transactions: {
    successful: {
      hash: '0xa1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456',
      from: '0x1234567890123456789012345678901234567890',
      to: '0x0987654321098765432109876543210987654321',
      value: '1000000000000000000',
      gasUsed: 21000,
      gasPrice: '20000000000',
      status: 'success',
      blockNumber: 12345,
      timestamp: '2024-01-01T00:00:00Z'
    },
    
    failed: {
      hash: '0xfedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321',
      from: '0x1234567890123456789012345678901234567890',
      to: '0x0987654321098765432109876543210987654321',
      value: '0',
      gasUsed: 0,
      gasPrice: '20000000000',
      status: 'failed',
      error: 'insufficient funds',
      blockNumber: 12346,
      timestamp: '2024-01-01T00:01:00Z'
    }
  },
  
  contracts: {
    agentRegistry: {
      address: '0x1111111111111111111111111111111111111111',
      abi: [
        {
          'name': 'registerAgent',
          'type': 'function',
          'inputs': [
            { 'name': 'agentId', 'type': 'string' },
            { 'name': 'agentData', 'type': 'string' }
          ]
        }
      ]
    }
  }
};

/**
 * API response mock data
 */
const apiResponses = {
  success: {
    agents: {
      list: {
        status: 200,
        data: {
          agents: [agents.reasoning, agents.search],
          total: 2,
          page: 1,
          limit: 10
        }
      },
      
      single: {
        status: 200,
        data: agents.reasoning
      }
    },
    
    users: {
      list: {
        status: 200,
        data: {
          users: [users.admin, users.regular],
          total: 2,
          page: 1,
          limit: 10
        }
      },
      
      single: {
        status: 200,
        data: users.regular
      }
    }
  },
  
  errors: {
    notFound: {
      status: 404,
      data: {
        error: 'Resource not found',
        message: 'The requested resource could not be found'
      }
    },
    
    unauthorized: {
      status: 401,
      data: {
        error: 'Unauthorized',
        message: 'Authentication required'
      }
    },
    
    forbidden: {
      status: 403,
      data: {
        error: 'Forbidden',
        message: 'Insufficient permissions'
      }
    },
    
    serverError: {
      status: 500,
      data: {
        error: 'Internal server error',
        message: 'An unexpected error occurred'
      }
    }
  }
};

/**
 * Test scenarios
 */
const scenarios = {
  // Complete workflow scenarios
  agentRegistration: {
    input: {
      name: 'Test Registration Agent',
      type: 'processing',
      capabilities: ['data_processing', 'validation']
    },
    expected: {
      status: 'active',
      id: expect.any(String),
      created: expect.any(String)
    }
  },
  
  userAuthentication: {
    input: {
      username: 'testuser',
      password: 'testpassword'
    },
    expected: {
      token: expect.any(String),
      user: expect.objectContaining({
        id: expect.any(String),
        username: 'testuser'
      })
    }
  }
};

module.exports = {
  agents,
  users,
  blockchain,
  apiResponses,
  scenarios
};