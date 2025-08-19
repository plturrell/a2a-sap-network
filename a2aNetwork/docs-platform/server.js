const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const path = require('path');
const fs = require('fs').promises;
const MarkdownIt = require('markdown-it');
const Prism = require('prismjs');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

const md = new MarkdownIt({
  highlight: function (str, lang) {
    if (lang && Prism.languages[lang]) {
      try {
        return Prism.highlight(str, Prism.languages[lang], lang);
      } catch (__) {}
    }
    return '';
  }
});

// Middleware
app.use(express.json());
app.use(express.static('dist'));
app.use('/api', require('./api/routes'));

// Documentation structure
const docsStructure = {
  'getting-started': {
    title: 'Getting Started',
    order: 1,
    sections: [
      'introduction',
      'installation',
      'first-agent',
      'concepts'
    ]
  },
  'agent-development': {
    title: 'Agent Development',
    order: 2,
    sections: [
      'creating-agents',
      'services',
      'communication',
      'error-handling'
    ]
  },
  'multi-agent-systems': {
    title: 'Multi-Agent Systems',
    order: 3,
    sections: [
      'workflows',
      'orchestration',
      'coordination',
      'patterns'
    ]
  },
  'blockchain-integration': {
    title: 'Blockchain Integration',
    order: 4,
    sections: [
      'setup',
      'smart-contracts',
      'reputation',
      'governance'
    ]
  },
  'enterprise-features': {
    title: 'Enterprise Features',
    order: 5,
    sections: [
      'sap-integration',
      'security',
      'monitoring',
      'deployment'
    ]
  },
  'api-reference': {
    title: 'API Reference',
    order: 6,
    sections: [
      'agent-api',
      'registry-api',
      'blockchain-api',
      'utilities'
    ]
  }
};

// Live code execution environment
const codeExecutor = require('./lib/CodeExecutor');
const exampleManager = require('./lib/ExampleManager');

// Socket connections for live features
io.on('connection', (socket) => {
  console.log('User connected to docs platform');

  // Handle live code execution
  socket.on('execute-code', async (data) => {
    try {
      const result = await codeExecutor.execute(data.code, data.language, data.context);
      socket.emit('code-result', { id: data.id, result });
    } catch (error) {
      socket.emit('code-error', { id: data.id, error: error.message });
    }
  });

  // Handle tutorial progress
  socket.on('tutorial-progress', (data) => {
    // Track user progress through tutorials
    console.log('Tutorial progress:', data);
  });

  // Handle live agent connections
  socket.on('connect-agent', async (data) => {
    try {
      const agentInfo = await exampleManager.connectToAgent(data.url);
      socket.emit('agent-connected', agentInfo);
    } catch (error) {
      socket.emit('agent-error', { error: error.message });
    }
  });

  socket.on('disconnect', () => {
    console.log('User disconnected from docs platform');
  });
});

// API Routes
app.get('/api/docs/structure', (req, res) => {
  res.json(docsStructure);
});

app.get('/api/docs/:category/:section', async (req, res) => {
  try {
    const { category, section } = req.params;
    const filePath = path.join(__dirname, 'content', category, `${section}.md`);
    const content = await fs.readFile(filePath, 'utf8');
    
    // Parse front matter and content
    const parsed = parseMarkdownWithMeta(content);
    
    res.json({
      content: md.render(parsed.content),
      meta: parsed.meta,
      interactive: parsed.interactive || []
    });
  } catch (error) {
    res.status(404).json({ error: 'Documentation not found' });
  }
});

app.get('/api/examples/:category', async (req, res) => {
  try {
    const { category } = req.params;
    const examples = await exampleManager.getExamples(category);
    res.json(examples);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/examples/run', async (req, res) => {
  try {
    const { code, language, context } = req.body;
    const result = await codeExecutor.execute(code, language, context);
    res.json({ result });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Serve the main application
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

function parseMarkdownWithMeta(content) {
  const lines = content.split('\n');
  let meta = {};
  let interactive = [];
  let contentStart = 0;

  // Parse YAML front matter
  if (lines[0] === '---') {
    const endIndex = lines.findIndex((line, index) => index > 0 && line === '---');
    if (endIndex > 0) {
      const yamlContent = lines.slice(1, endIndex).join('\n');
      try {
        const yaml = require('yaml');
        meta = yaml.parse(yamlContent);
      } catch (error) {
        console.error('Error parsing YAML front matter:', error);
      }
      contentStart = endIndex + 1;
    }
  }

  const markdownContent = lines.slice(contentStart).join('\n');

  // Extract interactive code blocks
  const interactiveRegex = /```interactive:(\w+)\s*\n([\s\S]*?)\n```/g;
  let match;
  while ((match = interactiveRegex.exec(markdownContent)) !== null) {
    interactive.push({
      language: match[1],
      code: match[2].trim(),
      id: `interactive-${interactive.length}`
    });
  }

  return {
    meta,
    content: markdownContent,
    interactive
  };
}

const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  console.log(`📚 A2A Documentation Platform running on port ${PORT}`);
  console.log(`🌐 Access at: http://localhost:${PORT}`);
});