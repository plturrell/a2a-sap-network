if (!process.addAsyncListener) require('../index.js');

const test = require('tap').test;
const net = require('net');

test('synchronous errors during connect return a null _handle', (t) =>{
  t.plan(3);

  // listening server
  const server = net.createServer().listen(8000);

  // client
  const client = net.connect({port: 8000});

  client.on('connect', () =>{
    t.ok(true, 'connect');
    // kill connection
    client.end();
  });

  client.on('error', () =>{
    server.close();
    t.ok(true, 'done test');
  });

  client.on('end', () => {
    setTimeout(() =>{
      // try to reconnect, but this has an error
      // rather than throw the right error, we're going to get an async-listener error
      t.ok(true, 'end');
      client.connect(8001);
    }, 100);
  });
});
