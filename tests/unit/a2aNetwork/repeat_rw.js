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

const testRW = require('../test_rw');
const test = require('tape');

const atoms = require('../atoms');
const RepeatRW = require('../repeat');
const SeriesRW = require('../series');
const ReadResult = require('../base').ReadResult;
const StructRW = require('../struct');

const brokenRW = {
    poolByteLength: function(destResult) {
        return destResult.reset(new Error('boom'));
    },
    poolWriteInto: function(destResult, val, buffer, offset) {
        return destResult.reset(new Error('bang'), offset);
    },
    poolReadFrom: function(destResult, buffer, offset) {
        return destResult.reset(new Error('bork'), offset);
    },
};

brokenRW.prototype = require('../base').BufferRW.prototype;

// n:1 (x<Int8>){n}
const tinyIntList = RepeatRW(atoms.UInt8, atoms.Int8);
test('RepeatRW: tinyIntList', testRW.cases(tinyIntList, [
    [[], [0x00]],
    [[-1, 0, 1], [0x03,
                  0xff,
                  0x00,
                  0x01]]
]));

// n:2 (x<Int16BE>){n}
const shortIntList = RepeatRW(atoms.UInt16BE, atoms.Int16BE);
test('RepeatRW: shortIntList', testRW.cases(shortIntList, [
    [[], [0x00, 0x00]],
    [[-1, 0, 1], [0x00, 0x03,
                  0xff, 0xff,
                  0x00, 0x00,
                  0x00, 0x01]],

    // invalid arguments through length/write
    {
        lengthTest: {
            value: 42,
            error: {
                type: 'bufrw.invalid-argument',
                message: 'invalid argument, expected an array',
                argType: 'number',
                argConstructor: 'Number'
            }
        },
        writeTest: {
            value: 42,
            error: {
                type: 'bufrw.invalid-argument',
                message: 'invalid argument, expected an array',
                argType: 'number',
                argConstructor: 'Number'
            }
        }
    }

]));

test('RepeatRW: passes countrw error thru', testRW.cases(RepeatRW(brokenRW, atoms.Int8), [
    {
        lengthTest: {
            value: [],
            error: {message: 'boom'}
        },
        writeTest: {
            value: [],
            length: 1,
            error: {message: 'bang'}
        },
        readTest: {
            bytes: [0],
            error: {message: 'bork'}
        }
    }
]));

test('RepeatRW: passes partrw error thru', testRW.cases(RepeatRW(atoms.UInt8, brokenRW), [
    {
        lengthTest: {
            value: [1],
            error: {message: 'boom'}
        },
        writeTest: {
            value: [1],
            length: 1,
            error: {message: 'bang'}
        },
        readTest: {
            bytes: [1, 1],
            error: {message: 'bork'}
        }
    }
]));

test('RepeatRW: properly handles repeated array rws', (assert) => {
    const thing = RepeatRW(atoms.UInt8, SeriesRW(atoms.UInt8, atoms.UInt8));
    const buf = Buffer.from([0x01, 0x02, 0x03]);

    const readResult = new ReadResult();
    thing.poolReadFrom(readResult, buf, 0);

    assert.deepEquals(readResult.value, [[2, 3]]);

    assert.end();
});

function Loc(lat, lng) {
    if (!(this instanceof Loc)) {
        return new Loc(lat, lng);
    }
    const self = this;
    self.lat = lat || 0;
    self.lng = lng || 0;
}

const consLoc = StructRW(Loc, {
    lat: atoms.DoubleBE,
    lng: atoms.DoubleBE
});

test('RepeatRW: properly handles repeated object rws', (assert) => {
    const thing = RepeatRW(atoms.UInt8, consLoc);
    const buf = Buffer.from([0x01, 0x40, 0x42, 0xe3, 0x43, 0x7c, 0x56, 0x92, 0xb4,
      0xc0, 0x5e, 0x9a, 0xb8, 0xa1, 0x9c, 0x9d, 0x5a]);

    const readResult = new ReadResult();
    thing.poolReadFrom(readResult, buf, 0);

    assert.deepEquals(readResult.value, [{lat: 37.775497, lng: -122.417519}]);

    assert.end();
});
