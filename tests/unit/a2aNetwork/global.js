'use strict';

const inspect = require('../');

const test = require('tape');
const globalThis = require('globalthis')();

test('global object', (t) => {
    /* eslint-env browser */
    const expected = typeof window === 'undefined' ? 'globalThis' : 'Window';
    t.equal(
        inspect([globalThis]),
        `[ { [object ${  expected  }] } ]`
    );

    t.end();
});
