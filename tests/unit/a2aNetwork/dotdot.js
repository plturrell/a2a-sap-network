const path = require('path');
const test = require('tape');
const resolve = require('../');

test('dotdot', (t) => {
    t.plan(4);
    const dir = path.join(__dirname, '/dotdot/abc');

    resolve('..', { basedir: dir }, (err, res, pkg) => {
        t.ifError(err);
        t.equal(res, path.join(__dirname, 'dotdot/index.js'));
    });

    resolve('.', { basedir: dir }, (err, res, pkg) => {
        t.ifError(err);
        t.equal(res, path.join(dir, 'index.js'));
    });
});

test('dotdot sync', (t) => {
    t.plan(2);
    const dir = path.join(__dirname, '/dotdot/abc');

    const a = resolve.sync('..', { basedir: dir });
    t.equal(a, path.join(__dirname, 'dotdot/index.js'));

    const b = resolve.sync('.', { basedir: dir });
    t.equal(b, path.join(dir, 'index.js'));
});
