const path = require('path');
const test = require('tape');
const resolve = require('../');

test('precedence', (t) => {
    t.plan(3);
    const dir = path.join(__dirname, 'precedence/aaa');

    resolve('./', { basedir: dir }, (err, res, pkg) => {
        t.ifError(err);
        t.equal(res, path.join(dir, 'index.js'));
        t.equal(pkg.name, 'resolve');
    });
});

test('./ should not load ${dir}.js', (t) => { // eslint-disable-line no-template-curly-in-string
    t.plan(1);
    const dir = path.join(__dirname, 'precedence/bbb');

    resolve('./', { basedir: dir }, (err, res, pkg) => {
        t.ok(err);
    });
});
