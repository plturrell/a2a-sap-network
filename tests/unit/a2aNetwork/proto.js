'use strict';

/* eslint no-proto: 0 */

const parse = require('../');
const test = require('tape');

test('proto pollution', (t) => {
	const argv = parse(['--__proto__.x', '123']);
	t.equal({}.x, undefined);
	t.equal(argv.__proto__.x, undefined);
	t.equal(argv.x, undefined);
	t.end();
});

test('proto pollution (array)', (t) => {
	const argv = parse(['--x', '4', '--x', '5', '--x.__proto__.z', '789']);
	t.equal({}.z, undefined);
	t.deepEqual(argv.x, [4, 5]);
	t.equal(argv.x.z, undefined);
	t.equal(argv.x.__proto__.z, undefined);
	t.end();
});

test('proto pollution (number)', (t) => {
	const argv = parse(['--x', '5', '--x.__proto__.z', '100']);
	t.equal({}.z, undefined);
	t.equal((4).z, undefined);
	t.equal(argv.x, 5);
	t.equal(argv.x.z, undefined);
	t.end();
});

test('proto pollution (string)', (t) => {
	const argv = parse(['--x', 'abc', '--x.__proto__.z', 'def']);
	t.equal({}.z, undefined);
	t.equal('...'.z, undefined);
	t.equal(argv.x, 'abc');
	t.equal(argv.x.z, undefined);
	t.end();
});

test('proto pollution (constructor)', (t) => {
	const argv = parse(['--constructor.prototype.y', '123']);
	t.equal({}.y, undefined);
	t.equal(argv.y, undefined);
	t.end();
});

test('proto pollution (constructor function)', (t) => {
	const argv = parse(['--_.concat.constructor.prototype.y', '123']);
	function fnToBeTested() {}
	t.equal(fnToBeTested.y, undefined);
	t.equal(argv.y, undefined);
	t.end();
});

// powered by snyk - https://github.com/backstage/backstage/issues/10343
test('proto pollution (constructor function) snyk', (t) => {
	const argv = parse('--_.constructor.constructor.prototype.foo bar'.split(' '));
	t.equal(function () {}.foo, undefined);
	t.equal(argv.y, undefined);
	t.end();
});
