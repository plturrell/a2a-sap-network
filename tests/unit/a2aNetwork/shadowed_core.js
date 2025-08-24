const test = require('tape');
const resolve = require('../');
const path = require('path');

test('shadowed core modules still return core module', (t) => {
    t.plan(2);

    resolve('util', { basedir: path.join(__dirname, 'shadowed_core') }, (err, res) => {
        t.ifError(err);
        t.equal(res, 'util');
    });
});

test('shadowed core modules still return core module [sync]', (t) => {
    t.plan(1);

    const res = resolve.sync('util', { basedir: path.join(__dirname, 'shadowed_core') });

    t.equal(res, 'util');
});

test('shadowed core modules return shadow when appending `/`', (t) => {
    t.plan(2);

    resolve('util/', { basedir: path.join(__dirname, 'shadowed_core') }, (err, res) => {
        t.ifError(err);
        t.equal(res, path.join(__dirname, 'shadowed_core/node_modules/util/index.js'));
    });
});

test('shadowed core modules return shadow when appending `/` [sync]', (t) => {
    t.plan(1);

    const res = resolve.sync('util/', { basedir: path.join(__dirname, 'shadowed_core') });

    t.equal(res, path.join(__dirname, 'shadowed_core/node_modules/util/index.js'));
});

test('shadowed core modules return shadow with `includeCoreModules: false`', (t) => {
    t.plan(2);

    resolve('util', { basedir: path.join(__dirname, 'shadowed_core'), includeCoreModules: false }, (err, res) => {
        t.ifError(err);
        t.equal(res, path.join(__dirname, 'shadowed_core/node_modules/util/index.js'));
    });
});

test('shadowed core modules return shadow with `includeCoreModules: false` [sync]', (t) => {
    t.plan(1);

    const res = resolve.sync('util', { basedir: path.join(__dirname, 'shadowed_core'), includeCoreModules: false });

    t.equal(res, path.join(__dirname, 'shadowed_core/node_modules/util/index.js'));
});
