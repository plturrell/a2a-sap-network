const path = require('path');
const test = require('tape');
const resolve = require('../');

test('moduleDirectory strings', (t) => {
    t.plan(4);
    const dir = path.join(__dirname, 'module_dir');
    const xopts = {
        basedir: dir,
        moduleDirectory: 'xmodules'
    };
    resolve('aaa', xopts, (err, res, pkg) => {
        t.ifError(err);
        t.equal(res, path.join(dir, '/xmodules/aaa/index.js'));
    });

    const yopts = {
        basedir: dir,
        moduleDirectory: 'ymodules'
    };
    resolve('aaa', yopts, (err, res, pkg) => {
        t.ifError(err);
        t.equal(res, path.join(dir, '/ymodules/aaa/index.js'));
    });
});

test('moduleDirectory array', (t) => {
    t.plan(6);
    const dir = path.join(__dirname, 'module_dir');
    const aopts = {
        basedir: dir,
        moduleDirectory: ['xmodules', 'ymodules', 'zmodules']
    };
    resolve('aaa', aopts, (err, res, pkg) => {
        t.ifError(err);
        t.equal(res, path.join(dir, '/xmodules/aaa/index.js'));
    });

    const bopts = {
        basedir: dir,
        moduleDirectory: ['zmodules', 'ymodules', 'xmodules']
    };
    resolve('aaa', bopts, (err, res, pkg) => {
        t.ifError(err);
        t.equal(res, path.join(dir, '/ymodules/aaa/index.js'));
    });

    const copts = {
        basedir: dir,
        moduleDirectory: ['xmodules', 'ymodules', 'zmodules']
    };
    resolve('bbb', copts, (err, res, pkg) => {
        t.ifError(err);
        t.equal(res, path.join(dir, '/zmodules/bbb/main.js'));
    });
});
