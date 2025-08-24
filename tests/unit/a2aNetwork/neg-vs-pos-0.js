const test = require('tape');
const equal = require('../');

test('0 values', (t) => {
    t.ok(equal( 0,  0), ' 0 ===  0');
    t.ok(equal( 0, +0), ' 0 === +0');
    t.ok(equal(+0, +0), '+0 === +0');
    t.ok(equal(-0, -0), '-0 === -0');

    t.notOk(equal(-0,  0), '-0 !==  0');
    t.notOk(equal(-0, +0), '-0 !== +0');

    t.end();
});

