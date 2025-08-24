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
const dns = require('dns');
const fs = require('fs');
const net = require('net');
const addListener = process.addAsyncListener;
const removeListener = process.removeAsyncListener;

let caught = 0;
let expectCaught = 0;

function asyncL() { }

const callbacksObj = {
  error: function(domain, er) {
    caught++;

    switch (er.message) {
      case 'sync throw':
      case 'setTimeout - simple':
      case 'setImmediate - simple':
      case 'setInterval - simple':
      case 'process.nextTick - simple':
      case 'setTimeout - nested':
      case 'process.nextTick - nested':
      case 'setImmediate - nested':
      case 'setTimeout2 - nested':
      case 'setInterval - nested':
      case 'fs - file does not exist':
      case 'fs - nested file does not exist':
      case 'fs - exists':
      case 'fs - realpath':
      case 'net - connection listener':
      case 'net - server listening':
      case 'net - client connect':
      case 'dns - lookup':
        return true;

      default:
        return false;
    }
  }
};

process.on('exit', () => {
  console.log('caught:', caught);
  console.log('expected:', expectCaught);
  assert.equal(caught, expectCaught, 'caught all expected errors');
  console.log('ok');
});

const listener = process.createAsyncListener(asyncL, callbacksObj);


// Catch synchronous throws
process.nextTick(() => {
  addListener(listener);

  expectCaught++;
  throw new Error('sync throw');

  removeListener(listener);
});


// Simple cases
process.nextTick(() => {
  addListener(listener);

  setTimeout(() => {
    throw new Error('setTimeout - simple');
  });
  expectCaught++;

  setImmediate(() => {
    throw new Error('setImmediate - simple');
  });
  expectCaught++;

  var b = setInterval(() => {
    clearInterval(b);
    throw new Error('setInterval - simple');
  });
  expectCaught++;

  process.nextTick(() => {
    throw new Error('process.nextTick - simple');
  });
  expectCaught++;

  removeListener(listener);
});


// Deeply nested
process.nextTick(() => {
  addListener(listener);

  setTimeout(() => {
    process.nextTick(() => {
      setImmediate(() => {
        var b = setInterval(() => {
          clearInterval(b);
          throw new Error('setInterval - nested');
        });
        expectCaught++;
        throw new Error('setImmediate - nested');
      });
      expectCaught++;
      throw new Error('process.nextTick - nested');
    });
    expectCaught++;
    setTimeout(() => {
      throw new Error('setTimeout2 - nested');
    });
    expectCaught++;
    throw new Error('setTimeout - nested');
  });
  expectCaught++;

  removeListener(listener);
});


// FS
process.nextTick(() => {
  addListener(listener);

  fs.stat('does not exist', () => {
    throw new Error('fs - file does not exist');
  });
  expectCaught++;

  fs.exists('hi all', () => {
    throw new Error('fs - exists');
  });
  expectCaught++;

  fs.realpath('/some/path', () => {
    throw new Error('fs - realpath');
  });
  expectCaught++;

  removeListener(listener);
});


// Nested FS
process.nextTick(() => {
  addListener(listener);

  setTimeout(() => {
    setImmediate(() => {
      var b = setInterval(() => {
        clearInterval(b);
        process.nextTick(() => {
          fs.stat('does not exist', () => {
            throw new Error('fs - nested file does not exist');
          });
          expectCaught++;
        });
      });
    });
  });

  removeListener(listener);
});


// Net
process.nextTick(() => {
  addListener(listener);

  var server = net.createServer(() => {
    server.close();
    throw new Error('net - connection listener');
  });
  expectCaught++;

  server.listen(PORT, () => {
    var client = net.connect(PORT, () => {
      client.end();
      throw new Error('net - client connect');
    });
    expectCaught++;
    throw new Error('net - server listening');
  });
  expectCaught++;

  removeListener(listener);
});


// DNS
process.nextTick(() => {
  addListener(listener);

  dns.lookup('localhost', () => {
    throw new Error('dns - lookup');
  });
  expectCaught++;

  removeListener(listener);
});
