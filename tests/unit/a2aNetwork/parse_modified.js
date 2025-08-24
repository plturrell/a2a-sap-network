'use strict';

const parse = require('../');
const test = require('tape');

test('parse with modifier functions', (t) => {
	t.plan(1);

	const argv = parse(['-b', '123'], { boolean: 'b' });
	t.deepEqual(argv, { b: true, _: [123] });
});
