const test = require('tape');
const table = require('../');

test('table', (t) => {
    t.plan(1);
    const s = table([
        [ 'master', '0123456789abcdef' ],
        [ 'staging', 'fedcba9876543210' ]
    ]);
    t.equal(s, [
        'master   0123456789abcdef',
        'staging  fedcba9876543210'
    ].join('\n'));
});
