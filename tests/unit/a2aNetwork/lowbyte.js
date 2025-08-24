const test = require('tape');
const inspect = require('../');

const obj = { x: 'a\r\nb', y: '\x05! \x1f \x12' };

test('interpolate low bytes', (t) => {
    t.plan(1);
    t.equal(
        inspect(obj),
        '{ x: \'a\\r\\nb\', y: \'\\x05! \\x1F \\x12\' }'
    );
});
