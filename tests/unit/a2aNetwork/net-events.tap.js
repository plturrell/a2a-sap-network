'use strict';

const net             = require('net')
  , tap             = require('tap')
  , test            = tap.test
  , createNamespace = require('../context').createNamespace
  ;

test('continuation-local state with net connection', (t) => {
  t.plan(4);

  const namespace = createNamespace('net');
  namespace.run(() => {
    namespace.set('test', 0xabad1dea);

    let server;
    namespace.run(() => {
      namespace.set('test', 0x1337);

      server = net.createServer((socket) => {
        t.equal(namespace.get('test'), 0x1337, 'state has been mutated');
        socket.on('data', () => {
          t.equal(namespace.get('test'), 0x1337, 'state is still preserved');
          server.close();
          socket.end('GoodBye');
        });
      });
      server.listen(() => {
        const address = server.address();
        namespace.run(() => {
          namespace.set('test', 'MONKEY');
          var client = net.connect(address.port, () => {
            t.equal(namespace.get('test'), 'MONKEY',
                    'state preserved for client connection');
            client.write('Hello');
            client.on('data', () => {
              t.equal(namespace.get('test'), 'MONKEY', 'state preserved for client data');
              t.end();
            });
          });
        });
      });
    });
  });
});
