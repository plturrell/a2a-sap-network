'use strict';

const parse = require('../');
const test = require('tape');

test('dotted alias', (t) => {
	const argv = parse(['--a.b', '22'], { default: { 'a.b': 11 }, alias: { 'a.b': 'aa.bb' } });
	t.equal(argv.a.b, 22);
	t.equal(argv.aa.bb, 22);
	t.end();
});

test('dotted default', (t) => {
	const argv = parse('', { default: { 'a.b': 11 }, alias: { 'a.b': 'aa.bb' } });
	t.equal(argv.a.b, 11);
	t.equal(argv.aa.bb, 11);
	t.end();
});

test('dotted default with no alias', (t) => {
	const argv = parse('', { default: { 'a.b': 11 } });
	t.equal(argv.a.b, 11);
	t.end();
});
