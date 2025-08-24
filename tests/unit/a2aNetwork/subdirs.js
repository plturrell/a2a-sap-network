const test = require('tape');
const resolve = require('../');
const path = require('path');

test('subdirs', (t) => {
    t.plan(2);

    const dir = path.join(__dirname, '/subdirs');
    resolve('a/b/c/x.json', { basedir: dir }, (err, res) => {
        t.ifError(err);
        t.equal(res, path.join(dir, 'node_modules/a/b/c/x.json'));
    });
});
