const test = require('tape');
const forEach = require('for-each');

const inspect = require('../');

test('bad indent options', (t) => {
    forEach([
        undefined,
        true,
        false,
        -1,
        1.2,
        Infinity,
        -Infinity,
        NaN
    ], (indent) => {
        t['throws'](
            () => { inspect('', { indent: indent }); },
            TypeError,
            `${inspect(indent)  } is invalid`
        );
    });

    t.end();
});

test('simple object with indent', (t) => {
    t.plan(2);

    const obj = { a: 1, b: 2 };

    const expectedSpaces = [
        '{',
        '  a: 1,',
        '  b: 2',
        '}'
    ].join('\n');
    const expectedTabs = [
        '{',
        '	a: 1,',
        '	b: 2',
        '}'
    ].join('\n');

    t.equal(inspect(obj, { indent: 2 }), expectedSpaces, 'two');
    t.equal(inspect(obj, { indent: '\t' }), expectedTabs, 'tabs');
});

test('two deep object with indent', (t) => {
    t.plan(2);

    const obj = { a: 1, b: { c: 3, d: 4 } };

    const expectedSpaces = [
        '{',
        '  a: 1,',
        '  b: {',
        '    c: 3,',
        '    d: 4',
        '  }',
        '}'
    ].join('\n');
    const expectedTabs = [
        '{',
        '	a: 1,',
        '	b: {',
        '		c: 3,',
        '		d: 4',
        '	}',
        '}'
    ].join('\n');

    t.equal(inspect(obj, { indent: 2 }), expectedSpaces, 'two');
    t.equal(inspect(obj, { indent: '\t' }), expectedTabs, 'tabs');
});

test('simple array with all single line elements', (t) => {
    t.plan(2);

    const obj = [1, 2, 3, 'asdf\nsdf'];

    const expected = '[ 1, 2, 3, \'asdf\\nsdf\' ]';

    t.equal(inspect(obj, { indent: 2 }), expected, 'two');
    t.equal(inspect(obj, { indent: '\t' }), expected, 'tabs');
});

test('array with complex elements', (t) => {
    t.plan(2);

    const obj = [1, { a: 1, b: { c: 1 } }, 'asdf\nsdf'];

    const expectedSpaces = [
        '[',
        '  1,',
        '  {',
        '    a: 1,',
        '    b: {',
        '      c: 1',
        '    }',
        '  },',
        '  \'asdf\\nsdf\'',
        ']'
    ].join('\n');
    const expectedTabs = [
        '[',
        '	1,',
        '	{',
        '		a: 1,',
        '		b: {',
        '			c: 1',
        '		}',
        '	},',
        '	\'asdf\\nsdf\'',
        ']'
    ].join('\n');

    t.equal(inspect(obj, { indent: 2 }), expectedSpaces, 'two');
    t.equal(inspect(obj, { indent: '\t' }), expectedTabs, 'tabs');
});

test('values', (t) => {
    t.plan(2);
    const obj = [{}, [], { 'a-b': 5 }];

    const expectedSpaces = [
        '[',
        '  {},',
        '  [],',
        '  {',
        '    \'a-b\': 5',
        '  }',
        ']'
    ].join('\n');
    const expectedTabs = [
        '[',
        '	{},',
        '	[],',
        '	{',
        '		\'a-b\': 5',
        '	}',
        ']'
    ].join('\n');

    t.equal(inspect(obj, { indent: 2 }), expectedSpaces, 'two');
    t.equal(inspect(obj, { indent: '\t' }), expectedTabs, 'tabs');
});

test('Map', { skip: typeof Map !== 'function' }, (t) => {
    const map = new Map();
    map.set({ a: 1 }, ['b']);
    map.set(3, NaN);

    const expectedStringSpaces = [
        'Map (2) {',
        '  { a: 1 } => [ \'b\' ],',
        '  3 => NaN',
        '}'
    ].join('\n');
    const expectedStringTabs = [
        'Map (2) {',
        '	{ a: 1 } => [ \'b\' ],',
        '	3 => NaN',
        '}'
    ].join('\n');
    const expectedStringTabsDoubleQuotes = [
        'Map (2) {',
        '	{ a: 1 } => [ "b" ],',
        '	3 => NaN',
        '}'
    ].join('\n');

    t.equal(
        inspect(map, { indent: 2 }),
        expectedStringSpaces,
        'Map keys are not indented (two)'
    );
    t.equal(
        inspect(map, { indent: '\t' }),
        expectedStringTabs,
        'Map keys are not indented (tabs)'
    );
    t.equal(
        inspect(map, { indent: '\t', quoteStyle: 'double' }),
        expectedStringTabsDoubleQuotes,
        'Map keys are not indented (tabs + double quotes)'
    );

    t.equal(inspect(new Map(), { indent: 2 }), 'Map (0) {}', 'empty Map should show as empty (two)');
    t.equal(inspect(new Map(), { indent: '\t' }), 'Map (0) {}', 'empty Map should show as empty (tabs)');

    const nestedMap = new Map();
    nestedMap.set(nestedMap, map);
    const expectedNestedSpaces = [
        'Map (1) {',
        '  [Circular] => Map (2) {',
        '    { a: 1 } => [ \'b\' ],',
        '    3 => NaN',
        '  }',
        '}'
    ].join('\n');
    const expectedNestedTabs = [
        'Map (1) {',
        '	[Circular] => Map (2) {',
        '		{ a: 1 } => [ \'b\' ],',
        '		3 => NaN',
        '	}',
        '}'
    ].join('\n');
    t.equal(inspect(nestedMap, { indent: 2 }), expectedNestedSpaces, 'Map containing a Map should work (two)');
    t.equal(inspect(nestedMap, { indent: '\t' }), expectedNestedTabs, 'Map containing a Map should work (tabs)');

    t.end();
});

test('Set', { skip: typeof Set !== 'function' }, (t) => {
    const set = new Set();
    set.add({ a: 1 });
    set.add(['b']);
    const expectedStringSpaces = [
        'Set (2) {',
        '  {',
        '    a: 1',
        '  },',
        '  [ \'b\' ]',
        '}'
    ].join('\n');
    const expectedStringTabs = [
        'Set (2) {',
        '	{',
        '		a: 1',
        '	},',
        '	[ \'b\' ]',
        '}'
    ].join('\n');
    t.equal(inspect(set, { indent: 2 }), expectedStringSpaces, 'new Set([{ a: 1 }, ["b"]]) should show size and contents (two)');
    t.equal(inspect(set, { indent: '\t' }), expectedStringTabs, 'new Set([{ a: 1 }, ["b"]]) should show size and contents (tabs)');

    t.equal(inspect(new Set(), { indent: 2 }), 'Set (0) {}', 'empty Set should show as empty (two)');
    t.equal(inspect(new Set(), { indent: '\t' }), 'Set (0) {}', 'empty Set should show as empty (tabs)');

    const nestedSet = new Set();
    nestedSet.add(set);
    nestedSet.add(nestedSet);
    const expectedNestedSpaces = [
        'Set (2) {',
        '  Set (2) {',
        '    {',
        '      a: 1',
        '    },',
        '    [ \'b\' ]',
        '  },',
        '  [Circular]',
        '}'
    ].join('\n');
    const expectedNestedTabs = [
        'Set (2) {',
        '	Set (2) {',
        '		{',
        '			a: 1',
        '		},',
        '		[ \'b\' ]',
        '	},',
        '	[Circular]',
        '}'
    ].join('\n');
    t.equal(inspect(nestedSet, { indent: 2 }), expectedNestedSpaces, 'Set containing a Set should work (two)');
    t.equal(inspect(nestedSet, { indent: '\t' }), expectedNestedTabs, 'Set containing a Set should work (tabs)');

    t.end();
});
