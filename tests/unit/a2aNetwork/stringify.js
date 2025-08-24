'use strict';

const test = require('tape');
const qs = require('../');
const utils = require('../lib/utils');
const iconv = require('iconv-lite');
const SaferBuffer = require('safer-buffer').Buffer;
const hasSymbols = require('has-symbols');
const hasBigInt = typeof BigInt === 'function';

test('stringify()', (t) => {
    t.test('stringifies a querystring object', (st) => {
        st.equal(qs.stringify({ a: 'b' }), 'a=b');
        st.equal(qs.stringify({ a: 1 }), 'a=1');
        st.equal(qs.stringify({ a: 1, b: 2 }), 'a=1&b=2');
        st.equal(qs.stringify({ a: 'A_Z' }), 'a=A_Z');
        st.equal(qs.stringify({ a: '€' }), 'a=%E2%82%AC');
        st.equal(qs.stringify({ a: '' }), 'a=%EE%80%80');
        st.equal(qs.stringify({ a: 'א' }), 'a=%D7%90');
        st.equal(qs.stringify({ a: '𐐷' }), 'a=%F0%90%90%B7');
        st.end();
    });

    t.test('stringifies falsy values', (st) => {
        st.equal(qs.stringify(undefined), '');
        st.equal(qs.stringify(null), '');
        st.equal(qs.stringify(null, { strictNullHandling: true }), '');
        st.equal(qs.stringify(false), '');
        st.equal(qs.stringify(0), '');
        st.end();
    });

    t.test('stringifies symbols', { skip: !hasSymbols() }, (st) => {
        st.equal(qs.stringify(Symbol.iterator), '');
        st.equal(qs.stringify([Symbol.iterator]), '0=Symbol%28Symbol.iterator%29');
        st.equal(qs.stringify({ a: Symbol.iterator }), 'a=Symbol%28Symbol.iterator%29');
        st.equal(
            qs.stringify({ a: [Symbol.iterator] }, { encodeValuesOnly: true, arrayFormat: 'brackets' }),
            'a[]=Symbol%28Symbol.iterator%29'
        );
        st.end();
    });

    t.test('stringifies bigints', { skip: !hasBigInt }, (st) => {
        const three = BigInt(3);
        const encodeWithN = function (value, defaultEncoder, charset) {
            const result = defaultEncoder(value, defaultEncoder, charset);
            return typeof value === 'bigint' ? `${result  }n` : result;
        };
        st.equal(qs.stringify(three), '');
        st.equal(qs.stringify([three]), '0=3');
        st.equal(qs.stringify([three], { encoder: encodeWithN }), '0=3n');
        st.equal(qs.stringify({ a: three }), 'a=3');
        st.equal(qs.stringify({ a: three }, { encoder: encodeWithN }), 'a=3n');
        st.equal(
            qs.stringify({ a: [three] }, { encodeValuesOnly: true, arrayFormat: 'brackets' }),
            'a[]=3'
        );
        st.equal(
            qs.stringify({ a: [three] }, { encodeValuesOnly: true, encoder: encodeWithN, arrayFormat: 'brackets' }),
            'a[]=3n'
        );
        st.end();
    });

    t.test('adds query prefix', (st) => {
        st.equal(qs.stringify({ a: 'b' }, { addQueryPrefix: true }), '?a=b');
        st.end();
    });

    t.test('with query prefix, outputs blank string given an empty object', (st) => {
        st.equal(qs.stringify({}, { addQueryPrefix: true }), '');
        st.end();
    });

    t.test('stringifies nested falsy values', (st) => {
        st.equal(qs.stringify({ a: { b: { c: null } } }), 'a%5Bb%5D%5Bc%5D=');
        st.equal(qs.stringify({ a: { b: { c: null } } }, { strictNullHandling: true }), 'a%5Bb%5D%5Bc%5D');
        st.equal(qs.stringify({ a: { b: { c: false } } }), 'a%5Bb%5D%5Bc%5D=false');
        st.end();
    });

    t.test('stringifies a nested object', (st) => {
        st.equal(qs.stringify({ a: { b: 'c' } }), 'a%5Bb%5D=c');
        st.equal(qs.stringify({ a: { b: { c: { d: 'e' } } } }), 'a%5Bb%5D%5Bc%5D%5Bd%5D=e');
        st.end();
    });

    t.test('stringifies a nested object with dots notation', (st) => {
        st.equal(qs.stringify({ a: { b: 'c' } }, { allowDots: true }), 'a.b=c');
        st.equal(qs.stringify({ a: { b: { c: { d: 'e' } } } }, { allowDots: true }), 'a.b.c.d=e');
        st.end();
    });

    t.test('stringifies an array value', (st) => {
        st.equal(
            qs.stringify({ a: ['b', 'c', 'd'] }, { arrayFormat: 'indices' }),
            'a%5B0%5D=b&a%5B1%5D=c&a%5B2%5D=d',
            'indices => indices'
        );
        st.equal(
            qs.stringify({ a: ['b', 'c', 'd'] }, { arrayFormat: 'brackets' }),
            'a%5B%5D=b&a%5B%5D=c&a%5B%5D=d',
            'brackets => brackets'
        );
        st.equal(
            qs.stringify({ a: ['b', 'c', 'd'] }, { arrayFormat: 'comma' }),
            'a=b%2Cc%2Cd',
            'comma => comma'
        );
        st.equal(
            qs.stringify({ a: ['b', 'c', 'd'] }),
            'a%5B0%5D=b&a%5B1%5D=c&a%5B2%5D=d',
            'default => indices'
        );
        st.end();
    });

    t.test('omits nulls when asked', (st) => {
        st.equal(qs.stringify({ a: 'b', c: null }, { skipNulls: true }), 'a=b');
        st.end();
    });

    t.test('omits nested nulls when asked', (st) => {
        st.equal(qs.stringify({ a: { b: 'c', d: null } }, { skipNulls: true }), 'a%5Bb%5D=c');
        st.end();
    });

    t.test('omits array indices when asked', (st) => {
        st.equal(qs.stringify({ a: ['b', 'c', 'd'] }, { indices: false }), 'a=b&a=c&a=d');
        st.end();
    });

    t.test('stringifies an array value with one item vs multiple items', (st) => {
        st.test('non-array item', (s2t) => {
            s2t.equal(qs.stringify({ a: 'c' }, { encodeValuesOnly: true, arrayFormat: 'indices' }), 'a=c');
            s2t.equal(qs.stringify({ a: 'c' }, { encodeValuesOnly: true, arrayFormat: 'brackets' }), 'a=c');
            s2t.equal(qs.stringify({ a: 'c' }, { encodeValuesOnly: true, arrayFormat: 'comma' }), 'a=c');
            s2t.equal(qs.stringify({ a: 'c' }, { encodeValuesOnly: true }), 'a=c');

            s2t.end();
        });

        st.test('array with a single item', (s2t) => {
            s2t.equal(qs.stringify({ a: ['c'] }, { encodeValuesOnly: true, arrayFormat: 'indices' }), 'a[0]=c');
            s2t.equal(qs.stringify({ a: ['c'] }, { encodeValuesOnly: true, arrayFormat: 'brackets' }), 'a[]=c');
            s2t.equal(qs.stringify({ a: ['c'] }, { encodeValuesOnly: true, arrayFormat: 'comma' }), 'a=c');
            s2t.equal(qs.stringify({ a: ['c'] }, { encodeValuesOnly: true, arrayFormat: 'comma', commaRoundTrip: true }), 'a[]=c'); // so it parses back as an array
            s2t.equal(qs.stringify({ a: ['c'] }, { encodeValuesOnly: true }), 'a[0]=c');

            s2t.end();
        });

        st.test('array with multiple items', (s2t) => {
            s2t.equal(qs.stringify({ a: ['c', 'd'] }, { encodeValuesOnly: true, arrayFormat: 'indices' }), 'a[0]=c&a[1]=d');
            s2t.equal(qs.stringify({ a: ['c', 'd'] }, { encodeValuesOnly: true, arrayFormat: 'brackets' }), 'a[]=c&a[]=d');
            s2t.equal(qs.stringify({ a: ['c', 'd'] }, { encodeValuesOnly: true, arrayFormat: 'comma' }), 'a=c,d');
            s2t.equal(qs.stringify({ a: ['c', 'd'] }, { encodeValuesOnly: true }), 'a[0]=c&a[1]=d');

            s2t.end();
        });

        st.end();
    });

    t.test('stringifies a nested array value', (st) => {
        st.equal(qs.stringify({ a: { b: ['c', 'd'] } }, { encodeValuesOnly: true, arrayFormat: 'indices' }), 'a[b][0]=c&a[b][1]=d');
        st.equal(qs.stringify({ a: { b: ['c', 'd'] } }, { encodeValuesOnly: true, arrayFormat: 'brackets' }), 'a[b][]=c&a[b][]=d');
        st.equal(qs.stringify({ a: { b: ['c', 'd'] } }, { encodeValuesOnly: true, arrayFormat: 'comma' }), 'a[b]=c,d');
        st.equal(qs.stringify({ a: { b: ['c', 'd'] } }, { encodeValuesOnly: true }), 'a[b][0]=c&a[b][1]=d');
        st.end();
    });

    t.test('stringifies a nested array value with dots notation', (st) => {
        st.equal(
            qs.stringify(
                { a: { b: ['c', 'd'] } },
                { allowDots: true, encodeValuesOnly: true, arrayFormat: 'indices' }
            ),
            'a.b[0]=c&a.b[1]=d',
            'indices: stringifies with dots + indices'
        );
        st.equal(
            qs.stringify(
                { a: { b: ['c', 'd'] } },
                { allowDots: true, encodeValuesOnly: true, arrayFormat: 'brackets' }
            ),
            'a.b[]=c&a.b[]=d',
            'brackets: stringifies with dots + brackets'
        );
        st.equal(
            qs.stringify(
                { a: { b: ['c', 'd'] } },
                { allowDots: true, encodeValuesOnly: true, arrayFormat: 'comma' }
            ),
            'a.b=c,d',
            'comma: stringifies with dots + comma'
        );
        st.equal(
            qs.stringify(
                { a: { b: ['c', 'd'] } },
                { allowDots: true, encodeValuesOnly: true }
            ),
            'a.b[0]=c&a.b[1]=d',
            'default: stringifies with dots + indices'
        );
        st.end();
    });

    t.test('stringifies an object inside an array', (st) => {
        st.equal(
            qs.stringify({ a: [{ b: 'c' }] }, { arrayFormat: 'indices' }),
            'a%5B0%5D%5Bb%5D=c', // a[0][b]=c
            'indices => brackets'
        );
        st.equal(
            qs.stringify({ a: [{ b: 'c' }] }, { arrayFormat: 'brackets' }),
            'a%5B%5D%5Bb%5D=c', // a[][b]=c
            'brackets => brackets'
        );
        st.equal(
            qs.stringify({ a: [{ b: 'c' }] }),
            'a%5B0%5D%5Bb%5D=c',
            'default => indices'
        );

        st.equal(
            qs.stringify({ a: [{ b: { c: [1] } }] }, { arrayFormat: 'indices' }),
            'a%5B0%5D%5Bb%5D%5Bc%5D%5B0%5D=1',
            'indices => indices'
        );

        st.equal(
            qs.stringify({ a: [{ b: { c: [1] } }] }, { arrayFormat: 'brackets' }),
            'a%5B%5D%5Bb%5D%5Bc%5D%5B%5D=1',
            'brackets => brackets'
        );

        st.equal(
            qs.stringify({ a: [{ b: { c: [1] } }] }),
            'a%5B0%5D%5Bb%5D%5Bc%5D%5B0%5D=1',
            'default => indices'
        );

        st.end();
    });

    t.test('stringifies an array with mixed objects and primitives', (st) => {
        st.equal(
            qs.stringify({ a: [{ b: 1 }, 2, 3] }, { encodeValuesOnly: true, arrayFormat: 'indices' }),
            'a[0][b]=1&a[1]=2&a[2]=3',
            'indices => indices'
        );
        st.equal(
            qs.stringify({ a: [{ b: 1 }, 2, 3] }, { encodeValuesOnly: true, arrayFormat: 'brackets' }),
            'a[][b]=1&a[]=2&a[]=3',
            'brackets => brackets'
        );
        st.equal(
            qs.stringify({ a: [{ b: 1 }, 2, 3] }, { encodeValuesOnly: true, arrayFormat: 'comma' }),
            '???',
            'brackets => brackets',
            { skip: 'TODO: figure out what this should do' }
        );
        st.equal(
            qs.stringify({ a: [{ b: 1 }, 2, 3] }, { encodeValuesOnly: true }),
            'a[0][b]=1&a[1]=2&a[2]=3',
            'default => indices'
        );

        st.end();
    });

    t.test('stringifies an object inside an array with dots notation', (st) => {
        st.equal(
            qs.stringify(
                { a: [{ b: 'c' }] },
                { allowDots: true, encode: false, arrayFormat: 'indices' }
            ),
            'a[0].b=c',
            'indices => indices'
        );
        st.equal(
            qs.stringify(
                { a: [{ b: 'c' }] },
                { allowDots: true, encode: false, arrayFormat: 'brackets' }
            ),
            'a[].b=c',
            'brackets => brackets'
        );
        st.equal(
            qs.stringify(
                { a: [{ b: 'c' }] },
                { allowDots: true, encode: false }
            ),
            'a[0].b=c',
            'default => indices'
        );

        st.equal(
            qs.stringify(
                { a: [{ b: { c: [1] } }] },
                { allowDots: true, encode: false, arrayFormat: 'indices' }
            ),
            'a[0].b.c[0]=1',
            'indices => indices'
        );
        st.equal(
            qs.stringify(
                { a: [{ b: { c: [1] } }] },
                { allowDots: true, encode: false, arrayFormat: 'brackets' }
            ),
            'a[].b.c[]=1',
            'brackets => brackets'
        );
        st.equal(
            qs.stringify(
                { a: [{ b: { c: [1] } }] },
                { allowDots: true, encode: false }
            ),
            'a[0].b.c[0]=1',
            'default => indices'
        );

        st.end();
    });

    t.test('does not omit object keys when indices = false', (st) => {
        st.equal(qs.stringify({ a: [{ b: 'c' }] }, { indices: false }), 'a%5Bb%5D=c');
        st.end();
    });

    t.test('uses indices notation for arrays when indices=true', (st) => {
        st.equal(qs.stringify({ a: ['b', 'c'] }, { indices: true }), 'a%5B0%5D=b&a%5B1%5D=c');
        st.end();
    });

    t.test('uses indices notation for arrays when no arrayFormat is specified', (st) => {
        st.equal(qs.stringify({ a: ['b', 'c'] }), 'a%5B0%5D=b&a%5B1%5D=c');
        st.end();
    });

    t.test('uses indices notation for arrays when no arrayFormat=indices', (st) => {
        st.equal(qs.stringify({ a: ['b', 'c'] }, { arrayFormat: 'indices' }), 'a%5B0%5D=b&a%5B1%5D=c');
        st.end();
    });

    t.test('uses repeat notation for arrays when no arrayFormat=repeat', (st) => {
        st.equal(qs.stringify({ a: ['b', 'c'] }, { arrayFormat: 'repeat' }), 'a=b&a=c');
        st.end();
    });

    t.test('uses brackets notation for arrays when no arrayFormat=brackets', (st) => {
        st.equal(qs.stringify({ a: ['b', 'c'] }, { arrayFormat: 'brackets' }), 'a%5B%5D=b&a%5B%5D=c');
        st.end();
    });

    t.test('stringifies a complicated object', (st) => {
        st.equal(qs.stringify({ a: { b: 'c', d: 'e' } }), 'a%5Bb%5D=c&a%5Bd%5D=e');
        st.end();
    });

    t.test('stringifies an empty value', (st) => {
        st.equal(qs.stringify({ a: '' }), 'a=');
        st.equal(qs.stringify({ a: null }, { strictNullHandling: true }), 'a');

        st.equal(qs.stringify({ a: '', b: '' }), 'a=&b=');
        st.equal(qs.stringify({ a: null, b: '' }, { strictNullHandling: true }), 'a&b=');

        st.equal(qs.stringify({ a: { b: '' } }), 'a%5Bb%5D=');
        st.equal(qs.stringify({ a: { b: null } }, { strictNullHandling: true }), 'a%5Bb%5D');
        st.equal(qs.stringify({ a: { b: null } }, { strictNullHandling: false }), 'a%5Bb%5D=');

        st.end();
    });

    t.test('stringifies an empty array in different arrayFormat', (st) => {
        st.equal(qs.stringify({ a: [], b: [null], c: 'c' }, { encode: false }), 'b[0]=&c=c');
        // arrayFormat default
        st.equal(qs.stringify({ a: [], b: [null], c: 'c' }, { encode: false, arrayFormat: 'indices' }), 'b[0]=&c=c');
        st.equal(qs.stringify({ a: [], b: [null], c: 'c' }, { encode: false, arrayFormat: 'brackets' }), 'b[]=&c=c');
        st.equal(qs.stringify({ a: [], b: [null], c: 'c' }, { encode: false, arrayFormat: 'repeat' }), 'b=&c=c');
        st.equal(qs.stringify({ a: [], b: [null], c: 'c' }, { encode: false, arrayFormat: 'comma' }), 'b=&c=c');
        st.equal(qs.stringify({ a: [], b: [null], c: 'c' }, { encode: false, arrayFormat: 'comma', commaRoundTrip: true }), 'b[]=&c=c');
        // with strictNullHandling
        st.equal(qs.stringify({ a: [], b: [null], c: 'c' }, { encode: false, arrayFormat: 'indices', strictNullHandling: true }), 'b[0]&c=c');
        st.equal(qs.stringify({ a: [], b: [null], c: 'c' }, { encode: false, arrayFormat: 'brackets', strictNullHandling: true }), 'b[]&c=c');
        st.equal(qs.stringify({ a: [], b: [null], c: 'c' }, { encode: false, arrayFormat: 'repeat', strictNullHandling: true }), 'b&c=c');
        st.equal(qs.stringify({ a: [], b: [null], c: 'c' }, { encode: false, arrayFormat: 'comma', strictNullHandling: true }), 'b&c=c');
        st.equal(qs.stringify({ a: [], b: [null], c: 'c' }, { encode: false, arrayFormat: 'comma', strictNullHandling: true, commaRoundTrip: true }), 'b[]&c=c');
        // with skipNulls
        st.equal(qs.stringify({ a: [], b: [null], c: 'c' }, { encode: false, arrayFormat: 'indices', skipNulls: true }), 'c=c');
        st.equal(qs.stringify({ a: [], b: [null], c: 'c' }, { encode: false, arrayFormat: 'brackets', skipNulls: true }), 'c=c');
        st.equal(qs.stringify({ a: [], b: [null], c: 'c' }, { encode: false, arrayFormat: 'repeat', skipNulls: true }), 'c=c');
        st.equal(qs.stringify({ a: [], b: [null], c: 'c' }, { encode: false, arrayFormat: 'comma', skipNulls: true }), 'c=c');

        st.end();
    });

    t.test('stringifies a null object', { skip: !Object.create }, (st) => {
        const obj = Object.create(null);
        obj.a = 'b';
        st.equal(qs.stringify(obj), 'a=b');
        st.end();
    });

    t.test('returns an empty string for invalid input', (st) => {
        st.equal(qs.stringify(undefined), '');
        st.equal(qs.stringify(false), '');
        st.equal(qs.stringify(null), '');
        st.equal(qs.stringify(''), '');
        st.end();
    });

    t.test('stringifies an object with a null object as a child', { skip: !Object.create }, (st) => {
        const obj = { a: Object.create(null) };

        obj.a.b = 'c';
        st.equal(qs.stringify(obj), 'a%5Bb%5D=c');
        st.end();
    });

    t.test('drops keys with a value of undefined', (st) => {
        st.equal(qs.stringify({ a: undefined }), '');

        st.equal(qs.stringify({ a: { b: undefined, c: null } }, { strictNullHandling: true }), 'a%5Bc%5D');
        st.equal(qs.stringify({ a: { b: undefined, c: null } }, { strictNullHandling: false }), 'a%5Bc%5D=');
        st.equal(qs.stringify({ a: { b: undefined, c: '' } }), 'a%5Bc%5D=');
        st.end();
    });

    t.test('url encodes values', (st) => {
        st.equal(qs.stringify({ a: 'b c' }), 'a=b%20c');
        st.end();
    });

    t.test('stringifies a date', (st) => {
        const now = new Date();
        const str = `a=${  encodeURIComponent(now.toISOString())}`;
        st.equal(qs.stringify({ a: now }), str);
        st.end();
    });

    t.test('stringifies the weird object from qs', (st) => {
        st.equal(qs.stringify({ 'my weird field': '~q1!2"\'w$5&7/z8)?' }), 'my%20weird%20field=~q1%212%22%27w%245%267%2Fz8%29%3F');
        st.end();
    });

    t.test('skips properties that are part of the object prototype', (st) => {
        Object.prototype.crash = 'test';
        st.equal(qs.stringify({ a: 'b' }), 'a=b');
        st.equal(qs.stringify({ a: { b: 'c' } }), 'a%5Bb%5D=c');
        delete Object.prototype.crash;
        st.end();
    });

    t.test('stringifies boolean values', (st) => {
        st.equal(qs.stringify({ a: true }), 'a=true');
        st.equal(qs.stringify({ a: { b: true } }), 'a%5Bb%5D=true');
        st.equal(qs.stringify({ b: false }), 'b=false');
        st.equal(qs.stringify({ b: { c: false } }), 'b%5Bc%5D=false');
        st.end();
    });

    t.test('stringifies buffer values', (st) => {
        st.equal(qs.stringify({ a: SaferBuffer.from('test') }), 'a=test');
        st.equal(qs.stringify({ a: { b: SaferBuffer.from('test') } }), 'a%5Bb%5D=test');
        st.end();
    });

    t.test('stringifies an object using an alternative delimiter', (st) => {
        st.equal(qs.stringify({ a: 'b', c: 'd' }, { delimiter: ';' }), 'a=b;c=d');
        st.end();
    });

    t.test('does not blow up when Buffer global is missing', (st) => {
        const tempBuffer = global.Buffer;
        delete global.Buffer;
        const result = qs.stringify({ a: 'b', c: 'd' });
        global.Buffer = tempBuffer;
        st.equal(result, 'a=b&c=d');
        st.end();
    });

    t.test('does not crash when parsing circular references', (st) => {
        const a = {};
        a.b = a;

        st['throws'](
            () => { qs.stringify({ 'foo[bar]': 'baz', 'foo[baz]': a }); },
            /RangeError: Cyclic object value/,
            'cyclic values throw'
        );

        const circular = {
            a: 'value'
        };
        circular.a = circular;
        st['throws'](
            () => { qs.stringify(circular); },
            /RangeError: Cyclic object value/,
            'cyclic values throw'
        );

        const arr = ['a'];
        st.doesNotThrow(
            () => { qs.stringify({ x: arr, y: arr }); },
            'non-cyclic values do not throw'
        );

        st.end();
    });

    t.test('non-circular duplicated references can still work', (st) => {
        const hourOfDay = {
            'function': 'hour_of_day'
        };

        const p1 = {
            'function': 'gte',
            arguments: [hourOfDay, 0]
        };
        const p2 = {
            'function': 'lte',
            arguments: [hourOfDay, 23]
        };

        st.equal(
            qs.stringify({ filters: { $and: [p1, p2] } }, { encodeValuesOnly: true }),
            'filters[$and][0][function]=gte&filters[$and][0][arguments][0][function]=hour_of_day&filters[$and][0][arguments][1]=0&filters[$and][1][function]=lte&filters[$and][1][arguments][0][function]=hour_of_day&filters[$and][1][arguments][1]=23'
        );

        st.end();
    });

    t.test('selects properties when filter=array', (st) => {
        st.equal(qs.stringify({ a: 'b' }, { filter: ['a'] }), 'a=b');
        st.equal(qs.stringify({ a: 1 }, { filter: [] }), '');

        st.equal(
            qs.stringify(
                { a: { b: [1, 2, 3, 4], c: 'd' }, c: 'f' },
                { filter: ['a', 'b', 0, 2], arrayFormat: 'indices' }
            ),
            'a%5Bb%5D%5B0%5D=1&a%5Bb%5D%5B2%5D=3',
            'indices => indices'
        );
        st.equal(
            qs.stringify(
                { a: { b: [1, 2, 3, 4], c: 'd' }, c: 'f' },
                { filter: ['a', 'b', 0, 2], arrayFormat: 'brackets' }
            ),
            'a%5Bb%5D%5B%5D=1&a%5Bb%5D%5B%5D=3',
            'brackets => brackets'
        );
        st.equal(
            qs.stringify(
                { a: { b: [1, 2, 3, 4], c: 'd' }, c: 'f' },
                { filter: ['a', 'b', 0, 2] }
            ),
            'a%5Bb%5D%5B0%5D=1&a%5Bb%5D%5B2%5D=3',
            'default => indices'
        );

        st.end();
    });

    t.test('supports custom representations when filter=function', (st) => {
        let calls = 0;
        const obj = { a: 'b', c: 'd', e: { f: new Date(1257894000000) } };
        const filterFunc = function (prefix, value) {
            calls += 1;
            if (calls === 1) {
                st.equal(prefix, '', 'prefix is empty');
                st.equal(value, obj);
            } else if (prefix === 'c') {
                return void 0;
            } else if (value instanceof Date) {
                st.equal(prefix, 'e[f]');
                return value.getTime();
            }
            return value;
        };

        st.equal(qs.stringify(obj, { filter: filterFunc }), 'a=b&e%5Bf%5D=1257894000000');
        st.equal(calls, 5);
        st.end();
    });

    t.test('can disable uri encoding', (st) => {
        st.equal(qs.stringify({ a: 'b' }, { encode: false }), 'a=b');
        st.equal(qs.stringify({ a: { b: 'c' } }, { encode: false }), 'a[b]=c');
        st.equal(qs.stringify({ a: 'b', c: null }, { strictNullHandling: true, encode: false }), 'a=b&c');
        st.end();
    });

    t.test('can sort the keys', (st) => {
        const sort = function (a, b) {
            return a.localeCompare(b);
        };
        st.equal(qs.stringify({ a: 'c', z: 'y', b: 'f' }, { sort: sort }), 'a=c&b=f&z=y');
        st.equal(qs.stringify({ a: 'c', z: { j: 'a', i: 'b' }, b: 'f' }, { sort: sort }), 'a=c&b=f&z%5Bi%5D=b&z%5Bj%5D=a');
        st.end();
    });

    t.test('can sort the keys at depth 3 or more too', (st) => {
        const sort = function (a, b) {
            return a.localeCompare(b);
        };
        st.equal(
            qs.stringify(
                { a: 'a', z: { zj: { zjb: 'zjb', zja: 'zja' }, zi: { zib: 'zib', zia: 'zia' } }, b: 'b' },
                { sort: sort, encode: false }
            ),
            'a=a&b=b&z[zi][zia]=zia&z[zi][zib]=zib&z[zj][zja]=zja&z[zj][zjb]=zjb'
        );
        st.equal(
            qs.stringify(
                { a: 'a', z: { zj: { zjb: 'zjb', zja: 'zja' }, zi: { zib: 'zib', zia: 'zia' } }, b: 'b' },
                { sort: null, encode: false }
            ),
            'a=a&z[zj][zjb]=zjb&z[zj][zja]=zja&z[zi][zib]=zib&z[zi][zia]=zia&b=b'
        );
        st.end();
    });

    t.test('can stringify with custom encoding', (st) => {
        st.equal(qs.stringify({ 県: '大阪府', '': '' }, {
            encoder: function (str) {
                if (str.length === 0) {
                    return '';
                }
                const buf = iconv.encode(str, 'shiftjis');
                const result = [];
                for (let i = 0; i < buf.length; ++i) {
                    result.push(buf.readUInt8(i).toString(16));
                }
                return `%${  result.join('%')}`;
            }
        }), '%8c%a7=%91%e5%8d%e3%95%7b&=');
        st.end();
    });

    t.test('receives the default encoder as a second argument', (st) => {
        st.plan(2);
        qs.stringify({ a: 1 }, {
            encoder: function (str, defaultEncoder) {
                st.equal(defaultEncoder, utils.encode);
            }
        });
        st.end();
    });

    t.test('throws error with wrong encoder', (st) => {
        st['throws'](() => {
            qs.stringify({}, { encoder: 'string' });
        }, new TypeError('Encoder has to be a function.'));
        st.end();
    });

    t.test('can use custom encoder for a buffer object', { skip: typeof Buffer === 'undefined' }, (st) => {
        st.equal(qs.stringify({ a: SaferBuffer.from([1]) }, {
            encoder: function (buffer) {
                if (typeof buffer === 'string') {
                    return buffer;
                }
                return String.fromCharCode(buffer.readUInt8(0) + 97);
            }
        }), 'a=b');

        st.equal(qs.stringify({ a: SaferBuffer.from('a b') }, {
            encoder: function (buffer) {
                return buffer;
            }
        }), 'a=a b');
        st.end();
    });

    t.test('serializeDate option', (st) => {
        const date = new Date();
        st.equal(
            qs.stringify({ a: date }),
            `a=${  date.toISOString().replace(/:/g, '%3A')}`,
            'default is toISOString'
        );

        const mutatedDate = new Date();
        mutatedDate.toISOString = function () {
            throw new SyntaxError();
        };
        st['throws'](() => {
            mutatedDate.toISOString();
        }, SyntaxError);
        st.equal(
            qs.stringify({ a: mutatedDate }),
            `a=${  Date.prototype.toISOString.call(mutatedDate).replace(/:/g, '%3A')}`,
            'toISOString works even when method is not locally present'
        );

        const specificDate = new Date(6);
        st.equal(
            qs.stringify(
                { a: specificDate },
                { serializeDate: function (d) { return d.getTime() * 7; } }
            ),
            'a=42',
            'custom serializeDate function called'
        );

        st.equal(
            qs.stringify(
                { a: [date] },
                {
                    serializeDate: function (d) { return d.getTime(); },
                    arrayFormat: 'comma'
                }
            ),
            `a=${  date.getTime()}`,
            'works with arrayFormat comma'
        );
        st.equal(
            qs.stringify(
                { a: [date] },
                {
                    serializeDate: function (d) { return d.getTime(); },
                    arrayFormat: 'comma',
                    commaRoundTrip: true
                }
            ),
            `a%5B%5D=${  date.getTime()}`,
            'works with arrayFormat comma'
        );

        st.end();
    });

    t.test('RFC 1738 serialization', (st) => {
        st.equal(qs.stringify({ a: 'b c' }, { format: qs.formats.RFC1738 }), 'a=b+c');
        st.equal(qs.stringify({ 'a b': 'c d' }, { format: qs.formats.RFC1738 }), 'a+b=c+d');
        st.equal(qs.stringify({ 'a b': SaferBuffer.from('a b') }, { format: qs.formats.RFC1738 }), 'a+b=a+b');

        st.equal(qs.stringify({ 'foo(ref)': 'bar' }, { format: qs.formats.RFC1738 }), 'foo(ref)=bar');

        st.end();
    });

    t.test('RFC 3986 spaces serialization', (st) => {
        st.equal(qs.stringify({ a: 'b c' }, { format: qs.formats.RFC3986 }), 'a=b%20c');
        st.equal(qs.stringify({ 'a b': 'c d' }, { format: qs.formats.RFC3986 }), 'a%20b=c%20d');
        st.equal(qs.stringify({ 'a b': SaferBuffer.from('a b') }, { format: qs.formats.RFC3986 }), 'a%20b=a%20b');

        st.end();
    });

    t.test('Backward compatibility to RFC 3986', (st) => {
        st.equal(qs.stringify({ a: 'b c' }), 'a=b%20c');
        st.equal(qs.stringify({ 'a b': SaferBuffer.from('a b') }), 'a%20b=a%20b');

        st.end();
    });

    t.test('Edge cases and unknown formats', (st) => {
        ['UFO1234', false, 1234, null, {}, []].forEach((format) => {
            st['throws'](
                () => {
                    qs.stringify({ a: 'b c' }, { format: format });
                },
                new TypeError('Unknown format option provided.')
            );
        });
        st.end();
    });

    t.test('encodeValuesOnly', (st) => {
        st.equal(
            qs.stringify(
                { a: 'b', c: ['d', 'e=f'], f: [['g'], ['h']] },
                { encodeValuesOnly: true }
            ),
            'a=b&c[0]=d&c[1]=e%3Df&f[0][0]=g&f[1][0]=h'
        );
        st.equal(
            qs.stringify(
                { a: 'b', c: ['d', 'e'], f: [['g'], ['h']] }
            ),
            'a=b&c%5B0%5D=d&c%5B1%5D=e&f%5B0%5D%5B0%5D=g&f%5B1%5D%5B0%5D=h'
        );
        st.end();
    });

    t.test('encodeValuesOnly - strictNullHandling', (st) => {
        st.equal(
            qs.stringify(
                { a: { b: null } },
                { encodeValuesOnly: true, strictNullHandling: true }
            ),
            'a[b]'
        );
        st.end();
    });

    t.test('throws if an invalid charset is specified', (st) => {
        st['throws'](() => {
            qs.stringify({ a: 'b' }, { charset: 'foobar' });
        }, new TypeError('The charset option must be either utf-8, iso-8859-1, or undefined'));
        st.end();
    });

    t.test('respects a charset of iso-8859-1', (st) => {
        st.equal(qs.stringify({ æ: 'æ' }, { charset: 'iso-8859-1' }), '%E6=%E6');
        st.end();
    });

    t.test('encodes unrepresentable chars as numeric entities in iso-8859-1 mode', (st) => {
        st.equal(qs.stringify({ a: '☺' }, { charset: 'iso-8859-1' }), 'a=%26%239786%3B');
        st.end();
    });

    t.test('respects an explicit charset of utf-8 (the default)', (st) => {
        st.equal(qs.stringify({ a: 'æ' }, { charset: 'utf-8' }), 'a=%C3%A6');
        st.end();
    });

    t.test('adds the right sentinel when instructed to and the charset is utf-8', (st) => {
        st.equal(qs.stringify({ a: 'æ' }, { charsetSentinel: true, charset: 'utf-8' }), 'utf8=%E2%9C%93&a=%C3%A6');
        st.end();
    });

    t.test('adds the right sentinel when instructed to and the charset is iso-8859-1', (st) => {
        st.equal(qs.stringify({ a: 'æ' }, { charsetSentinel: true, charset: 'iso-8859-1' }), 'utf8=%26%2310003%3B&a=%E6');
        st.end();
    });

    t.test('does not mutate the options argument', (st) => {
        const options = {};
        qs.stringify({}, options);
        st.deepEqual(options, {});
        st.end();
    });

    t.test('strictNullHandling works with custom filter', (st) => {
        const filter = function (prefix, value) {
            return value;
        };

        const options = { strictNullHandling: true, filter: filter };
        st.equal(qs.stringify({ key: null }, options), 'key');
        st.end();
    });

    t.test('strictNullHandling works with null serializeDate', (st) => {
        const serializeDate = function () {
            return null;
        };
        const options = { strictNullHandling: true, serializeDate: serializeDate };
        const date = new Date();
        st.equal(qs.stringify({ key: date }, options), 'key');
        st.end();
    });

    t.test('allows for encoding keys and values differently', (st) => {
        const encoder = function (str, defaultEncoder, charset, type) {
            if (type === 'key') {
                return defaultEncoder(str, defaultEncoder, charset, type).toLowerCase();
            }
            if (type === 'value') {
                return defaultEncoder(str, defaultEncoder, charset, type).toUpperCase();
            }
            throw `this should never happen! type: ${  type}`;
        };

        st.deepEqual(qs.stringify({ KeY: 'vAlUe' }, { encoder: encoder }), 'key=VALUE');
        st.end();
    });

    t.test('objects inside arrays', (st) => {
        const obj = { a: { b: { c: 'd', e: 'f' } } };
        const withArray = { a: { b: [{ c: 'd', e: 'f' }] } };

        st.equal(qs.stringify(obj, { encode: false }), 'a[b][c]=d&a[b][e]=f', 'no array, no arrayFormat');
        st.equal(qs.stringify(obj, { encode: false, arrayFormat: 'bracket' }), 'a[b][c]=d&a[b][e]=f', 'no array, bracket');
        st.equal(qs.stringify(obj, { encode: false, arrayFormat: 'indices' }), 'a[b][c]=d&a[b][e]=f', 'no array, indices');
        st.equal(qs.stringify(obj, { encode: false, arrayFormat: 'comma' }), 'a[b][c]=d&a[b][e]=f', 'no array, comma');

        st.equal(qs.stringify(withArray, { encode: false }), 'a[b][0][c]=d&a[b][0][e]=f', 'array, no arrayFormat');
        st.equal(qs.stringify(withArray, { encode: false, arrayFormat: 'bracket' }), 'a[b][0][c]=d&a[b][0][e]=f', 'array, bracket');
        st.equal(qs.stringify(withArray, { encode: false, arrayFormat: 'indices' }), 'a[b][0][c]=d&a[b][0][e]=f', 'array, indices');
        st.equal(
            qs.stringify(withArray, { encode: false, arrayFormat: 'comma' }),
            '???',
            'array, comma',
            { skip: 'TODO: figure out what this should do' }
        );

        st.end();
    });

    t.test('stringifies sparse arrays', (st) => {
        /* eslint no-sparse-arrays: 0 */
        st.equal(qs.stringify({ a: [, '2', , , '1'] }, { encodeValuesOnly: true }), 'a[1]=2&a[4]=1');
        st.equal(qs.stringify({ a: [, { b: [, , { c: '1' }] }] }, { encodeValuesOnly: true }), 'a[1][b][2][c]=1');
        st.equal(qs.stringify({ a: [, [, , [, , , { c: '1' }]]] }, { encodeValuesOnly: true }), 'a[1][2][3][c]=1');
        st.equal(qs.stringify({ a: [, [, , [, , , { c: [, '1'] }]]] }, { encodeValuesOnly: true }), 'a[1][2][3][c][1]=1');

        st.end();
    });

    t.end();
});
