const test = require('tape');
const path = require('path');
const resolve = require('../');

test('faulty basedir must produce error in windows', { skip: process.platform !== 'win32' }, (t) => {
    t.plan(1);

    const resolverDir = 'C:\\a\\b\\c\\d';

    resolve('tape/lib/test.js', { basedir: resolverDir }, (err, res, pkg) => {
        t.equal(!!err, true);
    });
});

test('non-existent basedir should not throw when preserveSymlinks is false', (t) => {
    t.plan(2);

    const opts = {
        basedir: path.join(path.sep, 'unreal', 'path', 'that', 'does', 'not', 'exist'),
        preserveSymlinks: false
    };

    const module = './dotdot/abc';

    resolve(module, opts, (err, res) => {
        t.equal(err.code, 'MODULE_NOT_FOUND');
        t.equal(res, undefined);
    });
});
