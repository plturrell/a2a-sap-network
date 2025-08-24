const inspect = require('../');
const test = require('tape');

test('deep', (t) => {
    t.plan(4);
    const obj = [[[[[[500]]]]]];
    t.equal(inspect(obj), '[ [ [ [ [ [Array] ] ] ] ] ]');
    t.equal(inspect(obj, { depth: 4 }), '[ [ [ [ [Array] ] ] ] ]');
    t.equal(inspect(obj, { depth: 2 }), '[ [ [Array] ] ]');

    t.equal(inspect([[[{ a: 1 }]]], { depth: 3 }), '[ [ [ [Object] ] ] ]');
});
