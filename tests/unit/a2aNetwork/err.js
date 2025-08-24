const test = require('tape');
const ErrorWithCause = require('error-cause/Error');

const inspect = require('../');

test('type error', (t) => {
    t.plan(1);
    const aerr = new TypeError();
    aerr.foo = 555;
    aerr.bar = [1, 2, 3];

    const berr = new TypeError('tuv');
    berr.baz = 555;

    const cerr = new SyntaxError();
    cerr.message = 'whoa';
    cerr['a-b'] = 5;

    const withCause = new ErrorWithCause('foo', { cause: 'bar' });
    const withCausePlus = new ErrorWithCause('foo', { cause: 'bar' });
    withCausePlus.foo = 'bar';
    const withUndefinedCause = new ErrorWithCause('foo', { cause: undefined });
    const withEnumerableCause = new Error('foo');
    withEnumerableCause.cause = 'bar';

    const obj = [
        new TypeError(),
        new TypeError('xxx'),
        aerr,
        berr,
        cerr,
        withCause,
        withCausePlus,
        withUndefinedCause,
        withEnumerableCause
    ];
    t.equal(inspect(obj), `[ ${  [
        '[TypeError]',
        '[TypeError: xxx]',
        '{ [TypeError] foo: 555, bar: [ 1, 2, 3 ] }',
        '{ [TypeError: tuv] baz: 555 }',
        '{ [SyntaxError: whoa] message: \'whoa\', \'a-b\': 5 }',
        'cause' in Error.prototype ? '[Error: foo]' : '{ [Error: foo] [cause]: \'bar\' }',
        `{ [Error: foo] ${  'cause' in Error.prototype ? '' : '[cause]: \'bar\', '  }foo: 'bar' }`,
        'cause' in Error.prototype ? '[Error: foo]' : '{ [Error: foo] [cause]: undefined }',
        '{ [Error: foo] cause: \'bar\' }'
    ].join(', ')  } ]`);
});
