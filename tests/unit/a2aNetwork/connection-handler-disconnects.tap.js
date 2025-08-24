'use strict';

const net = require('net');
const test = require('tap').test;
if (!process.addAsyncListener) require('../index.js');

const PORT = 12346;

test('another connection handler disconnects server', (t) => {
    t.plan(7);

    let client;

    // This tests that we don't crash when another connection listener
    // destroys the socket handle before we try to wrap
    // socket._handle.onread .
    // In this case, the connection handler declared below will run first,
    // because the wrapping event handler doesn't get added until
    // the server listens below.

    const server = net.createServer(() => {});
    server.on(
        'connection',
        (socket) => {
            t.ok(true, 'Reached second connection event');
            socket.destroy();
            t.ok(! socket._handle, 'Destroy removed the socket handle');
        }
    );

    server.on('error', (err) => {
        t.fail(true, 'It should not produce errors');
    });

    server.on(
        'listening',
        () => {
            t.ok(true, 'Server listened ok');

            // This will run both 'connection' handlers, with the one above
            // running first.
            // This should succeed even though the socket is destroyed.
            client = net.connect(PORT);
            client.on(
                'connect',
                () => {
                    t.ok(true, 'connected ok');
                }
            );

            client.on(
                'close',
                () => {
                    t.ok(true, 'disconnected ok');
                    t.ok(
                        !client._handle,
                        'close removed the client handle'
                    );

                    server.close(() => {
                        t.ok(
                            !server._handle,
                            'Destroy removed the server handle'
                        );
                    });
                }
            );
        }
    );

    // Another 'connection' handler is registered during this call.
    server.listen(PORT);

});
