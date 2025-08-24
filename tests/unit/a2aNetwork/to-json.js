'use strict';

const test = require('tape');
const stringify = require('../');

test('toJSON function', (t) => {
    t.plan(1);
    const obj = { one: 1, two: 2, toJSON: function() { return { one: 1 }; } };
    t.equal(stringify(obj), '{"one":1}' );
});

test('toJSON returns string', (t) => {
    t.plan(1);
    const obj = { one: 1, two: 2, toJSON: function() { return 'one'; } };
    t.equal(stringify(obj), '"one"');
});

test('toJSON returns array', (t) => {
    t.plan(1);
    const obj = { one: 1, two: 2, toJSON: function() { return ['one']; } };
    t.equal(stringify(obj), '["one"]');
});
