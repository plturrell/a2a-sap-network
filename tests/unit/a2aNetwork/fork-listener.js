'use strict';

const domain = require('domain');

if (!process.addAsyncListener) require('../index.js');

const d = domain.create();
d.on('error', (error) => {
  process.send(error.message);
});

process.on('message', (message) => {
  if (message === 'shutdown') {
    process.exit();
  }
  else {
    process.send(`child got unexpected message ${  message}`);
  }
});

d.run(() => {
  const server = require('net').createServer();

  server.on('error', () => {
    process.send('shutdown');
  });

  server.listen(8585, () => {
    process.send('child shouldn\'t be able to listen on port 8585');
  });
});
