'use strict';

const test = require('tape');
const parse = require('../');

test('boolean default true', (t) => {
	const argv = parse([], {
		boolean: 'sometrue',
		default: { sometrue: true },
	});
	t.equal(argv.sometrue, true);
	t.end();
});

test('boolean default false', (t) => {
	const argv = parse([], {
		boolean: 'somefalse',
		default: { somefalse: false },
	});
	t.equal(argv.somefalse, false);
	t.end();
});

test('boolean default to null', (t) => {
	const argv = parse([], {
		boolean: 'maybe',
		default: { maybe: null },
	});
	t.equal(argv.maybe, null);

	const argvLong = parse(['--maybe'], {
		boolean: 'maybe',
		default: { maybe: null },
	});
	t.equal(argvLong.maybe, true);
	t.end();
});
