module.exports = {
  ci: {
    collect: {
      url: [
        'http://localhost:8000',
        'http://localhost:8000/index.html'
      ],
      startServerCommand: 'cd a2aAgents/backend/app/a2a/developer_portal/static && python3 -m http.server 8000',
      startServerReadyPattern: 'Serving HTTP',
      startServerReadyTimeout: 30000
    },
    assert: {
      assertions: {
        'categories:performance': ['warn', {minScore: 0.8}],
        'categories:accessibility': ['error', {minScore: 0.9}],
        'categories:best-practices': ['warn', {minScore: 0.8}],
        'categories:seo': ['warn', {minScore: 0.8}]
      }
    },
    upload: {
      target: 'temporary-public-storage'
    }
  }
};
