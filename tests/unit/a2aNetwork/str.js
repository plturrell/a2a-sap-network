'use strict';

const test = require('tape');
const stringify = require('../');

test('simple object', (t) => {
    t.plan(1);
    const obj = { c: 6, b: [4,5], a: 3, z: null };
    t.equal(stringify(obj), '{"a":3,"b":[4,5],"c":6,"z":null}');
});

test('object with undefined', (t) => {
    t.plan(1);
    const obj = { a: 3, z: undefined };
    t.equal(stringify(obj), '{"a":3}');
});

test('object with null', (t) => {
    t.plan(1);
    const obj = { a: 3, z: null };
    t.equal(stringify(obj), '{"a":3,"z":null}');
});

test('object with NaN and Infinity', (t) => {
    t.plan(1);
    const obj = { a: 3, b: NaN, c: Infinity };
    t.equal(stringify(obj), '{"a":3,"b":null,"c":null}');
});

test('array with undefined', (t) => {
    t.plan(1);
    const obj = [4, undefined, 6];
    t.equal(stringify(obj), '[4,null,6]');
});

test('object with empty string', (t) => {
    t.plan(1);
    const obj = { a: 3, z: '' };
    t.equal(stringify(obj), '{"a":3,"z":""}');
});

test('array with empty string', (t) => {
    t.plan(1);
    const obj = [4, '', 6];
    t.equal(stringify(obj), '[4,"",6]');
});
