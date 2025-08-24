const test = require('tape');
const resolve = require('../');

test('nonstring', (t) => {
    t.plan(1);
    resolve(555, (err, res, pkg) => {
        t.ok(err);
    });
});
