const test = require('tape');
const stringify = require('../');

test('replace root', (t) => {
	t.plan(1);

	const obj = { a: 1, b: 2, c: false };
	const replacer = function(key, value) { return 'one'; };

	t.equal(stringify(obj, { replacer: replacer }), '"one"');
});

test('replace numbers', (t) => {
	t.plan(1);

	const obj = { a: 1, b: 2, c: false };
	const replacer = function(key, value) {
		if(value === 1) return 'one';
		if(value === 2) return 'two';
		return value;
	};

	t.equal(stringify(obj, { replacer: replacer }), '{"a":"one","b":"two","c":false}');
});

test('replace with object', (t) => {
	t.plan(1);

	const obj = { a: 1, b: 2, c: false };
	const replacer = function(key, value) {
		if(key === 'b') return { d: 1 };
		if(value === 1) return 'one';
		return value;
	};

	t.equal(stringify(obj, { replacer: replacer }), '{"a":"one","b":{"d":"one"},"c":false}');
});

test('replace with undefined', (t) => {
	t.plan(1);

	const obj = { a: 1, b: 2, c: false };
	const replacer = function(key, value) {
		if(value === false) return;
		return value;
	};

	t.equal(stringify(obj, { replacer: replacer }), '{"a":1,"b":2}');
});

test('replace with array', (t) => {
	t.plan(1);

	const obj = { a: 1, b: 2, c: false };
	const replacer = function(key, value) {
		if(key === 'b') return ['one', 'two'];
		return value;
	};

	t.equal(stringify(obj, { replacer: replacer }), '{"a":1,"b":["one","two"],"c":false}');
});

test('replace array item', (t) => {
	t.plan(1);

	const obj = { a: 1, b: 2, c: [1,2] };
	const replacer = function(key, value) {
		if(value === 1) return 'one';
		if(value === 2) return 'two';
		return value;
	};

	t.equal(stringify(obj, { replacer: replacer }), '{"a":"one","b":"two","c":["one","two"]}');
});
