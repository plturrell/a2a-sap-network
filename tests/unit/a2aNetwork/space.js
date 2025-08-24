const test = require('tape');
const stringify = require('../');

test('space parameter', (t) => {
    t.plan(1);
    const obj = { one: 1, two: 2 };
    t.equal(stringify(obj, {space: '  '}), ''
        + '{\n'
        + '  "one": 1,\n'
        + '  "two": 2\n'
        + '}'
    );
});

test('space parameter (with tabs)', (t) => {
    t.plan(1);
    const obj = { one: 1, two: 2 };
    t.equal(stringify(obj, {space: '\t'}), ''
        + '{\n'
        + '\t"one": 1,\n'
        + '\t"two": 2\n'
        + '}'
    );
});

test('space parameter (with a number)', (t) => {
    t.plan(1);
    const obj = { one: 1, two: 2 };
    t.equal(stringify(obj, {space: 3}), ''
        + '{\n'
        + '   "one": 1,\n'
        + '   "two": 2\n'
        + '}'
    );
});

test('space parameter (nested objects)', (t) => {
    t.plan(1);
    const obj = { one: 1, two: { b: 4, a: [2,3] } };
    t.equal(stringify(obj, {space: '  '}), ''
        + '{\n'
        + '  "one": 1,\n'
        + '  "two": {\n'
        + '    "a": [\n'
        + '      2,\n'
        + '      3\n'
        + '    ],\n'
        + '    "b": 4\n'
        + '  }\n'
        + '}'
    );
});

test('space parameter (same as native)', (t) => {
    t.plan(1);
    // for this test, properties need to be in alphabetical order
    const obj = { one: 1, two: { a: [2,3], b: 4 } };
    t.equal(stringify(obj, {space: '  '}), JSON.stringify(obj, null, '  '));
});
