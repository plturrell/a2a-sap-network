'use strict';

const test = require('tape');
const parse = require('../');

test('long opts', (t) => {
	t.deepEqual(
		parse(['--bool']),
		{ bool: true, _: [] },
		'long boolean'
	);
	t.deepEqual(
		parse(['--pow', 'xixxle']),
		{ pow: 'xixxle', _: [] },
		'long capture sp'
	);
	t.deepEqual(
		parse(['--pow=xixxle']),
		{ pow: 'xixxle', _: [] },
		'long capture eq'
	);
	t.deepEqual(
		parse(['--host', 'localhost', '--port', '555']),
		{ host: 'localhost', port: 555, _: [] },
		'long captures sp'
	);
	t.deepEqual(
		parse(['--host=localhost', '--port=555']),
		{ host: 'localhost', port: 555, _: [] },
		'long captures eq'
	);
	t.end();
});
