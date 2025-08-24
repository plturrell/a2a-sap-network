'use strict';

const fork = require('child_process').fork;
const test = require('tap').test;

let server;

test('parent listener', (t) => {
  server = require('net').createServer();

  server.listen(8585, () => {
    t.ok(server, 'parent listening on port 8585');

    const listener = fork(`${__dirname  }/fork-listener.js`);
    t.ok(listener, 'child process started');

    listener.on('message', (message) => {
      if (message === 'shutdown') {
        t.ok(message, 'child handled error properly');
        listener.send('shutdown');
      }
      else {
        t.fail(`parent got unexpected message ${  message}`);
      }
      t.end();
    });
  });
});

test('tearDown', (t) => {
  server.close();
  t.end();
});
