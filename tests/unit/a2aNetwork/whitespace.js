'use strict';

const parse = require('../');
const test = require('tape');

test('whitespace should be whitespace', (t) => {
	t.plan(1);
	const x = parse(['-x', '\t']).x;
	t.equal(x, '\t');
});
