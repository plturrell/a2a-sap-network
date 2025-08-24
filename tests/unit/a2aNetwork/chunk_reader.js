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

const PassThrough = require('stream').PassThrough;
const util = require('util');

const ChunkReader = require('../../stream/chunk_reader');

const testExpectations = require('../lib/test_expectations');
const byteLength = require('../../interface').byteLength;
const intoBuffer = require('../../interface').intoBuffer;
const UInt8 = require('../../atoms').UInt8;
const StringRW = require('../../string_rw');
const SeriesRW = require('../../series');
const BufferRW = require('../../base').BufferRW;

const str1 = StringRW(UInt8);
const frameRW = SeriesRW(UInt8, str1);
const readErrorRW = {
    poolReadFrom: function(destResult) {return destResult.reset(new Error('boom'), 0);},
    width: 1
};

readErrorRW.__proto__ = BufferRW.prototype;

const buffers = [];
const expectedFrames = [];
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
    const buf = intoBuffer(frameRW, Buffer.alloc(frame[0]), frame);
    buffers.push(buf);
    const assertMess = util.format('got expected[%s] payload token %j', i, token);
    expectedFrames.push({frame: expectToken});
    function expectToken(frame, assert) {
        assert.equal(frame[1], token, assertMess);
    }
});

const BigChunk = Buffer.concat(buffers);

const oneBytePer = new Array(BigChunk.length);
for (let i = 0; i < BigChunk.length; i++) {
    oneBytePer.push(Buffer.from([BigChunk[i]]));
}

readerTest('works frame-at-a-time', frameRW, buffers, expectedFrames);
readerTest('works from one big chunk', frameRW, [BigChunk], expectedFrames);
readerTest('works byte-at-a-time', frameRW, oneBytePer, expectedFrames);

function readerTest(desc, frameRW, chunks, expected) {
    testExpectations(desc, expected, (expect, done) => {
        const reader = ChunkReader(frameRW.rws[0], frameRW);
        const stream = PassThrough({highWaterMark: 1});
        chunks.forEach((chunk) => {
            stream.push(chunk);
        });
        stream.push(null);
        reader.on('data', (frame) => {
            expect('frame', frame);
        });
        reader.on('error', (err) => {
            expect('error', err);
            reader.end();
        });
        reader.on('finish', done);
        reader.on('end', done);
        stream.pipe(reader);
    });
}

readerTest('recognizes zero-sized chunks with an error', frameRW, [
    Buffer.from([0x00])
], [
    {
        error: function(err, assert) {
            assert.deepEqual(err, {
                fullType: 'bufrw.zero-length-chunk',
                type: 'bufrw.zero-length-chunk',
                name: 'BufrwZeroLengthChunkError',
                message: 'zero length chunk encountered'
            }, 'expected ZeroLengthChunkError');
        }
    }
]);

readerTest('handles sizeRW errors', SeriesRW(readErrorRW, str1), [
    Buffer.from([0x01])
], [
    {
        error: function(err, assert) {
            assert.equal(err.message, 'boom', 'expected boom');
        }
    }
]);

readerTest('handles chunkRW errors', SeriesRW(UInt8, readErrorRW), [
    Buffer.from([0x02, 0x00])
], [
    {
        error: function(err, assert) {
            assert.equal(err.message, 'boom', 'expected boom');
        }
    }
]);

readerTest('errors on truncated frame', frameRW, [
    Buffer.from([0x02, 0x01, 0x05])
], [
    {
        error: function(err, assert) {
            assert.deepEqual(err, {
                fullType: 'bufrw.short-buffer',
                type: 'bufrw.short-buffer',
                name: 'BufrwShortBufferError',
                message: 'expected at least 1 bytes, only have 0 @[1:2]',
                expected: 1,
                actual: 0,
                buffer: {
                    buffer: Buffer.from([2, 1]),
                    annotations: [
                        { kind: 'read', name: 'UInt8', start: 0, end: 1, value: 2 },
                        { kind: 'read', name: 'UInt8', start: 1, end: 2, value: 1 }
                    ]
                },
                offset: 1,
                endOffset: 2
            }, 'expected ShortBufferError');
        }
    },
    {
        error: function(err, assert) {
            assert.deepEqual(err, {
                fullType: 'bufrw.truncated-read',
                type: 'bufrw.truncated-read',
                name: 'BufrwTruncatedReadError',
                message: 'read truncated by end of stream with 1 bytes in buffer',
                state: 0,
                expecting: 4,
                length: 1,
                buffer: null
            }, 'expected TruncatedReadError');
        }
    }
]);
