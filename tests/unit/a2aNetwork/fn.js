const inspect = require('../');
const test = require('tape');
const arrow = require('make-arrow-function')();
const functionsHaveConfigurableNames = require('functions-have-names').functionsHaveConfigurableNames();

test('function', (t) => {
    t.plan(1);
    const obj = [1, 2, function f(n) { return n; }, 4];
    t.equal(inspect(obj), '[ 1, 2, [Function: f], 4 ]');
});

test('function name', (t) => {
    t.plan(1);
    const f = (function () {
        return function () {};
    }());
    f.toString = function toStr() { return 'function xxx () {}'; };
    const obj = [1, 2, f, 4];
    t.equal(inspect(obj), '[ 1, 2, [Function (anonymous)] { toString: [Function: toStr] }, 4 ]');
});

test('anon function', (t) => {
    const f = (function () {
        return function () {};
    }());
    const obj = [1, 2, f, 4];
    t.equal(inspect(obj), '[ 1, 2, [Function (anonymous)], 4 ]');

    t.end();
});

test('arrow function', { skip: !arrow }, (t) => {
    t.equal(inspect(arrow), '[Function (anonymous)]');

    t.end();
});

test('truly nameless function', { skip: !arrow || !functionsHaveConfigurableNames }, (t) => {
    function f() {}
    Object.defineProperty(f, 'name', { value: false });
    t.equal(f.name, false);
    t.equal(
        inspect(f),
        '[Function: f]',
        'named function with falsy `.name` does not hide its original name'
    );

    function g() {}
    Object.defineProperty(g, 'name', { value: true });
    t.equal(g.name, true);
    t.equal(
        inspect(g),
        '[Function: true]',
        'named function with truthy `.name` hides its original name'
    );

    const anon = function () {}; // eslint-disable-line func-style
    Object.defineProperty(anon, 'name', { value: null });
    t.equal(anon.name, null);
    t.equal(
        inspect(anon),
        '[Function (anonymous)]',
        'anon function with falsy `.name` does not hide its anonymity'
    );

    const anon2 = function () {}; // eslint-disable-line func-style
    Object.defineProperty(anon2, 'name', { value: 1 });
    t.equal(anon2.name, 1);
    t.equal(
        inspect(anon2),
        '[Function: 1]',
        'anon function with truthy `.name` hides its anonymity'
    );

    t.end();
});
