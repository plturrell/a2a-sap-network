// Copyright Joyent, Inc. and other Node contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the
// following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
// NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.

const PORT = 12346;

if (!process.addAsyncListener) require('../index.js');
if (!global.setImmediate) global.setImmediate = setTimeout;

const assert = require('assert');
const net = require('net');
const fs = require('fs');
const dgram = require('dgram');

const addListener = process.addAsyncListener;
const removeListener = process.removeAsyncListener;
let actualAsync = 0;
let expectAsync = 0;

const callbacks = {
  create: function onAsync() {
    actualAsync++;
  }
};

const listener = process.createAsyncListener(callbacks);

process.on('exit', () => {
  console.log('expected', expectAsync);
  console.log('actual  ', actualAsync);
  // TODO(trevnorris): Not a great test. If one was missed, but others
  // overflowed then the test would still pass.
  assert.ok(actualAsync >= expectAsync);
});


// Test listeners side-by-side
process.nextTick(() => {
  addListener(listener);

  var b = setInterval(() => {
    clearInterval(b);
  });
  expectAsync++;

  var c = setInterval(() => {
    clearInterval(c);
  });
  expectAsync++;

  setTimeout(() => { });
  expectAsync++;

  setTimeout(() => { });
  expectAsync++;

  process.nextTick(() => { });
  expectAsync++;

  process.nextTick(() => { });
  expectAsync++;

  setImmediate(() => { });
  expectAsync++;

  setImmediate(() => { });
  expectAsync++;

  setTimeout(() => { }, 10);
  expectAsync++;

  setTimeout(() => { }, 10);
  expectAsync++;

  removeListener(listener);
});


// Async listeners should propagate with nested callbacks
process.nextTick(() => {
  addListener(listener);
  let interval = 3;

  process.nextTick(() => {
    setTimeout(() => {
      setImmediate(() => {
        var i = setInterval(() => {
          if (--interval <= 0)
            clearInterval(i);
        });
        expectAsync++;
      });
      expectAsync++;
      process.nextTick(() => {
        setImmediate(() => {
          setTimeout(() => { }, 20);
          expectAsync++;
        });
        expectAsync++;
      });
      expectAsync++;
    });
    expectAsync++;
  });
  expectAsync++;

  removeListener(listener);
});


// Test triggers with two async listeners
process.nextTick(() => {
  addListener(listener);
  addListener(listener);

  setTimeout(() => {
    process.nextTick(() => { });
    expectAsync += 2;
  });
  expectAsync += 2;

  removeListener(listener);
  removeListener(listener);
});


// Test callbacks from fs I/O
process.nextTick(() => {
  addListener(listener);

  fs.stat('something random', (err, stat) => { });
  expectAsync++;

  setImmediate(() => {
    fs.stat('random again', (err, stat) => { });
    expectAsync++;
  });
  expectAsync++;

  removeListener(listener);
});


// Test net I/O
process.nextTick(() => {
  addListener(listener);

  const server = net.createServer((c) => { });
  expectAsync++;

  server.listen(PORT, () => {
    server.close();
    expectAsync++;
  });
  expectAsync++;

  removeListener(listener);
});


// Test UDP
process.nextTick(() => {
  addListener(listener);

  const server = dgram.createSocket('udp4');
  expectAsync++;

  server.bind(PORT);

  server.close();
  expectAsync++;

  removeListener(listener);
});
