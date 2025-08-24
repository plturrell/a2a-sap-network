'use strict';

const parse = require('../');
const test = require('tape');

test('stops parsing on the first non-option when stopEarly is set', (t) => {
	const argv = parse(['--aaa', 'bbb', 'ccc', '--ddd'], {
		stopEarly: true,
	});

	t.deepEqual(argv, {
		aaa: 'bbb',
		_: ['ccc', '--ddd'],
	});

	t.end();
});
