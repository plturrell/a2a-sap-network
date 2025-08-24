'use strict';

const parse = require('../');
const test = require('tape');

test('flag boolean default false', (t) => {
	const argv = parse(['moo'], {
		boolean: ['t', 'verbose'],
		default: { verbose: false, t: false },
	});

	t.deepEqual(argv, {
		verbose: false,
		t: false,
		_: ['moo'],
	});

	t.deepEqual(typeof argv.verbose, 'boolean');
	t.deepEqual(typeof argv.t, 'boolean');
	t.end();

});

test('boolean groups', (t) => {
	const argv = parse(['-x', '-z', 'one', 'two', 'three'], {
		boolean: ['x', 'y', 'z'],
	});

	t.deepEqual(argv, {
		x: true,
		y: false,
		z: true,
		_: ['one', 'two', 'three'],
	});

	t.deepEqual(typeof argv.x, 'boolean');
	t.deepEqual(typeof argv.y, 'boolean');
	t.deepEqual(typeof argv.z, 'boolean');
	t.end();
});
test('boolean and alias with chainable api', (t) => {
	const aliased = ['-h', 'derp'];
	const regular = ['--herp', 'derp'];
	const aliasedArgv = parse(aliased, {
		boolean: 'herp',
		alias: { h: 'herp' },
	});
	const propertyArgv = parse(regular, {
		boolean: 'herp',
		alias: { h: 'herp' },
	});
	const expected = {
		herp: true,
		h: true,
		_: ['derp'],
	};

	t.same(aliasedArgv, expected);
	t.same(propertyArgv, expected);
	t.end();
});

test('boolean and alias with options hash', (t) => {
	const aliased = ['-h', 'derp'];
	const regular = ['--herp', 'derp'];
	const opts = {
		alias: { h: 'herp' },
		boolean: 'herp',
	};
	const aliasedArgv = parse(aliased, opts);
	const propertyArgv = parse(regular, opts);
	const expected = {
		herp: true,
		h: true,
		_: ['derp'],
	};
	t.same(aliasedArgv, expected);
	t.same(propertyArgv, expected);
	t.end();
});

test('boolean and alias array with options hash', (t) => {
	const aliased = ['-h', 'derp'];
	const regular = ['--herp', 'derp'];
	const alt = ['--harp', 'derp'];
	const opts = {
		alias: { h: ['herp', 'harp'] },
		boolean: 'h',
	};
	const aliasedArgv = parse(aliased, opts);
	const propertyArgv = parse(regular, opts);
	const altPropertyArgv = parse(alt, opts);
	const expected = {
		harp: true,
		herp: true,
		h: true,
		_: ['derp'],
	};
	t.same(aliasedArgv, expected);
	t.same(propertyArgv, expected);
	t.same(altPropertyArgv, expected);
	t.end();
});

test('boolean and alias using explicit true', (t) => {
	const aliased = ['-h', 'true'];
	const regular = ['--herp', 'true'];
	const opts = {
		alias: { h: 'herp' },
		boolean: 'h',
	};
	const aliasedArgv = parse(aliased, opts);
	const propertyArgv = parse(regular, opts);
	const expected = {
		herp: true,
		h: true,
		_: [],
	};

	t.same(aliasedArgv, expected);
	t.same(propertyArgv, expected);
	t.end();
});

// regression, see https://github.com/substack/node-optimist/issues/71
test('boolean and --x=true', (t) => {
	let parsed = parse(['--boool', '--other=true'], {
		boolean: 'boool',
	});

	t.same(parsed.boool, true);
	t.same(parsed.other, 'true');

	parsed = parse(['--boool', '--other=false'], {
		boolean: 'boool',
	});

	t.same(parsed.boool, true);
	t.same(parsed.other, 'false');
	t.end();
});

test('boolean --boool=true', (t) => {
	const parsed = parse(['--boool=true'], {
		default: {
			boool: false,
		},
		boolean: ['boool'],
	});

	t.same(parsed.boool, true);
	t.end();
});

test('boolean --boool=false', (t) => {
	const parsed = parse(['--boool=false'], {
		default: {
			boool: true,
		},
		boolean: ['boool'],
	});

	t.same(parsed.boool, false);
	t.end();
});

test('boolean using something similar to true', (t) => {
	const opts = { boolean: 'h' };
	const result = parse(['-h', 'true.txt'], opts);
	const expected = {
		h: true,
		_: ['true.txt'],
	};

	t.same(result, expected);
	t.end();
});
