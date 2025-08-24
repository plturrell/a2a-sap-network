// Copyright (c) 2015 Uber Technologies, Inc.

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

const PassThrough = require('stream').PassThrough;
const util = require('util');

const ChunkWriter = require('../../stream/chunk_writer.js');

const testExpectations = require('../lib/test_expectations');
const byteLength = require('../../interface').byteLength;
const intoBuffer = require('../../interface').intoBuffer;
const UInt8 = require('../../atoms').UInt8;
const StringRW = require('../../string_rw');
const SeriesRW = require('../../series');
const BufferRW = require('../../base').BufferRW;

const str1 = StringRW(UInt8);
const frameRW = SeriesRW(UInt8, str1);
const writeErrorRW = {
    poolByteLength: function(destResult) {
        return destResult.reset(null, 0);
    },
    poolWriteInto: function(destResult) {
        return destResult.reset(new Error('boom'), 0);
    }
};

writeErrorRW.__proto__ = BufferRW.prototype;

const frames = [];
const expectedBuffers = [];
[
    'boot', 'cat',
    'boots', 'cats',
    'boots', 'N',
    'cats', 'N',
    'boots', 'N',
    'cats'
].forEach((token, i) => {
    const frame = [0, token];
    frame[0] = byteLength(frameRW, frame);
    const expectedBuffer = intoBuffer(frameRW, Buffer.alloc(frame[0]), frame);
    const assertMess = util.format('got expected[%s] buffer', i);
    frames.push(frame);
    expectedBuffers.push({
        buffer: function expectToken(buffer, assert) {
            assert.deepEqual(buffer, expectedBuffer, assertMess);
        }
    });
});

function writerTest(desc, frameRW, frames, expected) {
    testExpectations(desc, expected, (expect, done) => {
        const writer = ChunkWriter(frameRW);
        const stream = PassThrough({
            objectMode: true
        });
        frames.forEach((frame) => {
            stream.push(frame);
        });
        stream.push(null);
        writer.on('data', (buffer) => {
            expect('buffer', buffer);
        });
        writer.on('error', (err) => {
            expect('error', err);
            writer.end();
        });
        writer.on('finish', done);
        writer.on('end', done);
        stream.pipe(writer);
    });
}

writerTest('writes expected frame buffers', frameRW, frames, expectedBuffers);

writerTest('handles write errors', SeriesRW(UInt8, writeErrorRW),
    [[1, '']],
    [
        {
            error: function(err, assert) {
                assert.equal(err.message, 'boom while writing [ 1, \'\' ]', 'expected boom');
            }
        }
    ]);
