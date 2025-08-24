const test = require('tape');
const equal = require('../');

test('NaN and 0 values', (t) => {
    t.ok(equal(NaN, NaN));
    t.notOk(equal(0, NaN));
    t.ok(equal(0, 0));
    t.notOk(equal(0, 1));
    t.end();
});


test('nested NaN values', (t) => {
    t.ok(equal([ NaN, 1, NaN ], [ NaN, 1, NaN ]));
    t.end();
});
