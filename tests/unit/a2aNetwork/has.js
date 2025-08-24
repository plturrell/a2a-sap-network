'use strict';

const inspect = require('../');
const test = require('tape');
const mockProperty = require('mock-property');

test('when Object#hasOwnProperty is deleted', (t) => {
    t.plan(1);
    const arr = [1, , 3]; // eslint-disable-line no-sparse-arrays

    t.teardown(mockProperty(Array.prototype, 1, { value: 2 })); // this is needed to account for "in" vs "hasOwnProperty"
    t.teardown(mockProperty(Object.prototype, 'hasOwnProperty', { 'delete': true }));

    t.equal(inspect(arr), '[ 1, , 3 ]');
});
