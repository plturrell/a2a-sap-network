const test = require('tape');
const v = require('es-value-fixtures');
const forEach = require('for-each');

const inspect = require('../');

test('negative zero', (t) => {
    t.equal(inspect(0), '0', 'inspect(0) === "0"');
    t.equal(inspect(Object(0)), 'Object(0)', 'inspect(Object(0)) === "Object(0)"');

    t.equal(inspect(-0), '-0', 'inspect(-0) === "-0"');
    t.equal(inspect(Object(-0)), 'Object(-0)', 'inspect(Object(-0)) === "Object(-0)"');

    t.end();
});

test('numericSeparator', (t) => {
    forEach(v.nonBooleans, (nonBoolean) => {
        t['throws'](
            () => { inspect(true, { numericSeparator: nonBoolean }); },
            TypeError,
            `${inspect(nonBoolean)  } is not a boolean`
        );
    });

    t.test('3 digit numbers', (st) => {
        let failed = false;
        for (let i = -999; i < 1000; i += 1) {
            const actual = inspect(i);
            const actualSepNo = inspect(i, { numericSeparator: false });
            const actualSepYes = inspect(i, { numericSeparator: true });
            const expected = String(i);
            if (actual !== expected || actualSepNo !== expected || actualSepYes !== expected) {
                failed = true;
                t.equal(actual, expected);
                t.equal(actualSepNo, expected);
                t.equal(actualSepYes, expected);
            }
        }

        st.notOk(failed, 'all 3 digit numbers passed');

        st.end();
    });

    t.equal(inspect(1e3), '1000', '1000');
    t.equal(inspect(1e3, { numericSeparator: false }), '1000', '1000, numericSeparator false');
    t.equal(inspect(1e3, { numericSeparator: true }), '1_000', '1000, numericSeparator true');
    t.equal(inspect(-1e3), '-1000', '-1000');
    t.equal(inspect(-1e3, { numericSeparator: false }), '-1000', '-1000, numericSeparator false');
    t.equal(inspect(-1e3, { numericSeparator: true }), '-1_000', '-1000, numericSeparator true');

    t.equal(inspect(1234.5678, { numericSeparator: true }), '1_234.567_8', 'fractional numbers get separators');
    t.equal(inspect(1234.56789, { numericSeparator: true }), '1_234.567_89', 'fractional numbers get separators');
    t.equal(inspect(1234.567891, { numericSeparator: true }), '1_234.567_891', 'fractional numbers get separators');

    t.end();
});
