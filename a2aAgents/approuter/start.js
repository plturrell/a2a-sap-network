const approuter = require('@sap/approuter');

// Custom middleware for additional security and monitoring
const customMiddleware = {
  first: [
    // Request ID middleware
    (req, res, next) => {
      const requestId = req.headers['x-request-id'] || 
                       require('crypto').randomUUID();
      req.requestId = requestId;
      res.setHeader('X-Request-ID', requestId);
      next();
    },
    
    // Security headers middleware (additional to built-in)
    (req, res, next) => {
      res.setHeader('X-Powered-By', ''); // Remove default header
      res.setHeader('Server', ''); // Remove server information
      next();
    },
    
    // Request logging middleware
    (req, res, next) => {
      // const start = Date.now();
      const originalSend = res.send;
      
      res.send = function(body) {
        // const duration = Date.now() - start;
        // Request logging - commented for production
        // console.log(JSON.stringify({
        //   timestamp: new Date().toISOString(),
        //   requestId: req.requestId,
        //   method: req.method,
        //   url: req.url,
        //   statusCode: res.statusCode,
        //   duration: duration,
        //   userAgent: req.headers['user-agent'],
        //   ip: req.headers['x-forwarded-for'] || req.connection.remoteAddress
        // }));
        originalSend.call(this, body);
      };
      
      next();
    }
  ]
};

// Start approuter with custom middleware
const ar = approuter({
  middleware: customMiddleware
});

ar.start({
  port: process.env.PORT || 5000
});