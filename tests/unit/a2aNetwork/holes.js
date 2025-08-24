const test = require('tape');
const inspect = require('../');

const xs = ['a', 'b'];
xs[5] = 'f';
xs[7] = 'j';
xs[8] = 'k';

test('holes', (t) => {
    t.plan(1);
    t.equal(
        inspect(xs),
        '[ \'a\', \'b\', , , , \'f\', , \'j\', \'k\' ]'
    );
});
