'use strict';

const test = require('tape');
const stringify = require('../');

test('custom comparison function', (t) => {
    t.plan(1);
    const obj = { c: 8, b: [{z:6,y:5,x:4},7], a: 3 };
    const s = stringify(obj, (a, b) => {
        return a.key < b.key ? 1 : -1;
    });
    t.equal(s, '{"c":8,"b":[{"z":6,"y":5,"x":4},7],"a":3}');
});
