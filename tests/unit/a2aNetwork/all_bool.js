'use strict';

const parse = require('../');
const test = require('tape');

test('flag boolean true (default all --args to boolean)', (t) => {
	const argv = parse(['moo', '--honk', 'cow'], {
		boolean: true,
	});

	t.deepEqual(argv, {
		honk: true,
		_: ['moo', 'cow'],
	});

	t.deepEqual(typeof argv.honk, 'boolean');
	t.end();
});

test('flag boolean true only affects double hyphen arguments without equals signs', (t) => {
	const argv = parse(['moo', '--honk', 'cow', '-p', '55', '--tacos=good'], {
		boolean: true,
	});

	t.deepEqual(argv, {
		honk: true,
		tacos: 'good',
		p: 55,
		_: ['moo', 'cow'],
	});

	t.deepEqual(typeof argv.honk, 'boolean');
	t.end();
});
