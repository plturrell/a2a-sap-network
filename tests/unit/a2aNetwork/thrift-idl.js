// Copyright (c) 2018 Uber Technologies, Inc.
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

/* eslint max-len:[0, 120] */
'use strict';

const idl = require('../thrift-idl');
const fs = require('fs');
const path = require('path');
const test = require('tape');

test('thrift IDL parser can parse thrift test files', (assert) => {
    const extension = '.thrift';
    const dirname = path.join(__dirname, 'thrift');
    const filenames = fs.readdirSync(dirname);
    for (let index = 0; index < filenames.length; index++) {
        const filename = filenames[index];
        const fullFilename = path.join(dirname, filename);
        if (filename.indexOf(extension, filename.length - extension.length) > 0) {
            const source = fs.readFileSync(fullFilename, 'ascii');
            try {
                idl.parse(source);
                assert.pass(filename);
            } catch (err) {
                assert.fail(filename);
            }
        }
    }
    assert.end();
});
