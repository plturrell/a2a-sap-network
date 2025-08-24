// Copyright (c) 2015 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

'use strict';

const test = require('tape');

const iface = require('../interface');
const BufferRW = require('../base').BufferRW;

const byteRW = {
    poolByteLength: function(destResult) {return destResult.reset(null, 1);},
    poolWriteInto: function(destResult, b, buffer, offset) {
        buffer[offset] = b;
        return destResult.reset(null, ++offset);
    },
    poolReadFrom: function(destResult, buffer, offset) {
        const b = buffer[offset];
        return destResult.reset(null, ++offset, b);
    },
};

byteRW.__proto__ = BufferRW.prototype;

const lengthErrorRW = {
    poolByteLength: function(destResult) {return destResult.reset(new Error('boom'));}
};

lengthErrorRW.__proto__ = BufferRW.prototype;

const writeErrorRW = {
    poolByteLength: function(destResult) {return destResult.reset(null, 0);},
    poolWriteInto: function(destResult) {return destResult.reset(new Error('bang'));}
};

writeErrorRW.__proto__ = BufferRW.prototype;

const readErrorRW = {
    poolReadFrom: function(destResult) {return destResult.reset(new Error('zot'));}
};

readErrorRW.__proto__ = BufferRW.prototype;

test('byteLength', (assert) => {
    assert.deepEqual(
        iface.byteLength(byteRW, 1),
        1, 'write 1 uint8');
    assert.throws(() => {
        iface.byteLength(lengthErrorRW, 1);
    }, /boom/, 'length error throws');
    assert.end();
});

test('toBuffer', (assert) => {
    assert.deepEqual(
        iface.toBuffer(byteRW, 1),
        Buffer.from([0x01]), 'write 1 uint8');
    assert.throws(() => {
        iface.toBuffer(lengthErrorRW, 1);
    }, /boom/, 'length error throws');
    assert.throws(() => {
        iface.toBuffer(writeErrorRW, 1);
    }, /bang/, 'write error throws');
    assert.end();
});

test('intoBuffer', (assert) => {
    assert.deepEqual(
        iface.intoBuffer(byteRW, Buffer.from([0]), 1),
        Buffer.from([0x01]), 'write 1 uint8');
    assert.throws(() => {
        iface.intoBuffer(writeErrorRW, Buffer.from([0]), 1);
    }, /bang/, 'write error throws');
    assert.throws(() => {
        iface.intoBuffer(byteRW, Buffer.from([0, 0]), 1);
    }, /short write, 1 byte left over after writing 1/, 'short write error');
    assert.end();
});

test('fromBuffer', (assert) => {
    assert.equal(
        iface.fromBuffer(byteRW, Buffer.from([0x01])),
        1, 'read 1 uint8');
    assert.throws(() => {
        iface.fromBuffer(readErrorRW, Buffer.alloc(0));
    }, /zot/, 'read error throws');
    assert.throws(() => {
        iface.fromBuffer(byteRW, Buffer.from([0, 0]));
    }, /short read, 1 byte left over after consuming 1/, 'short read error');
    assert.end();
});
