const test = require('tape');
const inspect = require('../');

const obj = { a: 1, b: [3, 4, undefined, null], c: undefined, d: null };

test('undef and null', (t) => {
    t.plan(1);
    t.equal(
        inspect(obj),
        '{ a: 1, b: [ 3, 4, undefined, null ], c: undefined, d: null }'
    );
});
