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


if (!process.addAsyncListener) require('../index.js');

const assert = require('assert');
const net = require('net');
const fs = require('fs');

let actualAsync = 0;
let expectAsync = 0;


process.on('exit', () => {
  console.log('expected', expectAsync);
  console.log('actual  ', actualAsync);
  assert.equal(expectAsync, actualAsync);
  console.log('ok');
});


// --- Begin Testing --- //

function onAsync() {
  actualAsync++;
}


process.addAsyncListener(onAsync);


// Test listeners side-by-side
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

setTimeout(() => { }, 100);
expectAsync++;

setTimeout(() => { }, 100);
expectAsync++;


// Async listeners should propagate with nested callbacks
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
        setTimeout(() => { }, 200);
        expectAsync++;
      });
      expectAsync++;
    });
    expectAsync++;
  });
  expectAsync++;
});
expectAsync++;


// Test callbacks from fs I/O
fs.stat('something random', () => { });
expectAsync++;

setImmediate(() => {
  fs.stat('random again', () => { });
  expectAsync++;
});
expectAsync++;


// Test net I/O
const server = net.createServer(() => { });
expectAsync++;

server.listen(8080, () => {
  server.close();
  expectAsync++;
});
expectAsync++;
