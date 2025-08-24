'use strict';

const GetIntrinsic = require('../');

const test = require('tape');
const forEach = require('for-each');
const debug = require('object-inspect');
const generatorFns = require('make-generator-function')();
const asyncFns = require('make-async-function').list();
const asyncGenFns = require('make-async-generator-function')();
const mockProperty = require('mock-property');

const callBound = require('call-bound');
const v = require('es-value-fixtures');
const $gOPD = require('gopd');
const DefinePropertyOrThrow = require('es-abstract/2023/DefinePropertyOrThrow');

const $isProto = callBound('%Object.prototype.isPrototypeOf%');

test('export', (t) => {
	t.equal(typeof GetIntrinsic, 'function', 'it is a function');
	t.equal(GetIntrinsic.length, 2, 'function has length of 2');

	t.end();
});

test('throws', (t) => {
	t['throws'](
		() => { GetIntrinsic('not an intrinsic'); },
		SyntaxError,
		'nonexistent intrinsic throws a syntax error'
	);

	t['throws'](
		() => { GetIntrinsic(''); },
		TypeError,
		'empty string intrinsic throws a type error'
	);

	t['throws'](
		() => { GetIntrinsic('.'); },
		SyntaxError,
		'"just a dot" intrinsic throws a syntax error'
	);

	t['throws'](
		() => { GetIntrinsic('%String'); },
		SyntaxError,
		'Leading % without trailing % throws a syntax error'
	);

	t['throws'](
		() => { GetIntrinsic('String%'); },
		SyntaxError,
		'Trailing % without leading % throws a syntax error'
	);

	t['throws'](
		() => { GetIntrinsic('String[\'prototype]'); },
		SyntaxError,
		'Dynamic property access is disallowed for intrinsics (unterminated string)'
	);

	t['throws'](
		() => { GetIntrinsic('%Proxy.prototype.undefined%'); },
		TypeError,
		'Throws when middle part doesn\'t exist (%Proxy.prototype.undefined%)'
	);

	t['throws'](
		() => { GetIntrinsic('%Array.prototype%garbage%'); },
		SyntaxError,
		'Throws with extra percent signs'
	);

	t['throws'](
		() => { GetIntrinsic('%Array.prototype%push%'); },
		SyntaxError,
		'Throws with extra percent signs, even on an existing intrinsic'
	);

	forEach(v.nonStrings, (nonString) => {
		t['throws'](
			() => { GetIntrinsic(nonString); },
			TypeError,
			`${debug(nonString)  } is not a String`
		);
	});

	forEach(v.nonBooleans, (nonBoolean) => {
		t['throws'](
			() => { GetIntrinsic('%', nonBoolean); },
			TypeError,
			`${debug(nonBoolean)  } is not a Boolean`
		);
	});

	forEach([
		'toString',
		'propertyIsEnumerable',
		'hasOwnProperty'
	], (objectProtoMember) => {
		t['throws'](
			() => { GetIntrinsic(objectProtoMember); },
			SyntaxError,
			`${debug(objectProtoMember)  } is not an intrinsic`
		);
	});

	t.end();
});

test('base intrinsics', (t) => {
	t.equal(GetIntrinsic('%Object%'), Object, '%Object% yields Object');
	t.equal(GetIntrinsic('Object'), Object, 'Object yields Object');
	t.equal(GetIntrinsic('%Array%'), Array, '%Array% yields Array');
	t.equal(GetIntrinsic('Array'), Array, 'Array yields Array');

	t.end();
});

test('dotted paths', (t) => {
	t.equal(GetIntrinsic('%Object.prototype.toString%'), Object.prototype.toString, '%Object.prototype.toString% yields Object.prototype.toString');
	t.equal(GetIntrinsic('Object.prototype.toString'), Object.prototype.toString, 'Object.prototype.toString yields Object.prototype.toString');
	t.equal(GetIntrinsic('%Array.prototype.push%'), Array.prototype.push, '%Array.prototype.push% yields Array.prototype.push');
	t.equal(GetIntrinsic('Array.prototype.push'), Array.prototype.push, 'Array.prototype.push yields Array.prototype.push');

	test('underscore paths are aliases for dotted paths', { skip: !Object.isFrozen || Object.isFrozen(Object.prototype) }, (st) => {
		const original = GetIntrinsic('%ObjProto_toString%');

		forEach([
			'%Object.prototype.toString%',
			'Object.prototype.toString',
			'%ObjectPrototype.toString%',
			'ObjectPrototype.toString',
			'%ObjProto_toString%',
			'ObjProto_toString'
		], (name) => {
			DefinePropertyOrThrow(Object.prototype, 'toString', {
				'[[Value]]': function toString() {
					return original.apply(this, arguments);
				}
			});
			st.equal(GetIntrinsic(name), original, `${name  } yields original Object.prototype.toString`);
		});

		DefinePropertyOrThrow(Object.prototype, 'toString', { '[[Value]]': original });
		st.end();
	});

	test('dotted paths cache', { skip: !Object.isFrozen || Object.isFrozen(Object.prototype) }, (st) => {
		const original = GetIntrinsic('%Object.prototype.propertyIsEnumerable%');

		forEach([
			'%Object.prototype.propertyIsEnumerable%',
			'Object.prototype.propertyIsEnumerable',
			'%ObjectPrototype.propertyIsEnumerable%',
			'ObjectPrototype.propertyIsEnumerable'
		], (name) => {
			const restore = mockProperty(Object.prototype, 'propertyIsEnumerable', {
				value: function propertyIsEnumerable() {
					return original.apply(this, arguments);
				}
			});
			st.equal(GetIntrinsic(name), original, `${name  } yields cached Object.prototype.propertyIsEnumerable`);

			restore();
		});

		st.end();
	});

	test('dotted path reports correct error', (st) => {
		st['throws'](() => {
			GetIntrinsic('%NonExistentIntrinsic.prototype.property%');
		}, /%NonExistentIntrinsic%/, 'The base intrinsic of %NonExistentIntrinsic.prototype.property% is %NonExistentIntrinsic%');

		st['throws'](() => {
			GetIntrinsic('%NonExistentIntrinsicPrototype.property%');
		}, /%NonExistentIntrinsicPrototype%/, 'The base intrinsic of %NonExistentIntrinsicPrototype.property% is %NonExistentIntrinsicPrototype%');

		st.end();
	});

	t.end();
});

test('accessors', { skip: !$gOPD || typeof Map !== 'function' }, (t) => {
	const actual = $gOPD(Map.prototype, 'size');
	t.ok(actual, 'Map.prototype.size has a descriptor');
	t.equal(typeof actual.get, 'function', 'Map.prototype.size has a getter function');
	t.equal(GetIntrinsic('%Map.prototype.size%'), actual.get, '%Map.prototype.size% yields the getter for it');
	t.equal(GetIntrinsic('Map.prototype.size'), actual.get, 'Map.prototype.size yields the getter for it');

	t.end();
});

test('generator functions', { skip: !generatorFns.length }, (t) => {
	const $GeneratorFunction = GetIntrinsic('%GeneratorFunction%');
	const $GeneratorFunctionPrototype = GetIntrinsic('%Generator%');
	const $GeneratorPrototype = GetIntrinsic('%GeneratorPrototype%');

	forEach(generatorFns, (genFn) => {
		let fnName = genFn.name;
		fnName = fnName ? `'${  fnName  }'` : 'genFn';

		t.ok(genFn instanceof $GeneratorFunction, `${fnName  } instanceof %GeneratorFunction%`);
		t.ok($isProto($GeneratorFunctionPrototype, genFn), `%Generator% is prototype of ${  fnName}`);
		t.ok($isProto($GeneratorPrototype, genFn.prototype), `%GeneratorPrototype% is prototype of ${  fnName  }.prototype`);
	});

	t.end();
});

test('async functions', { skip: !asyncFns.length }, (t) => {
	const $AsyncFunction = GetIntrinsic('%AsyncFunction%');
	const $AsyncFunctionPrototype = GetIntrinsic('%AsyncFunctionPrototype%');

	forEach(asyncFns, (asyncFn) => {
		let fnName = asyncFn.name;
		fnName = fnName ? `'${  fnName  }'` : 'asyncFn';

		t.ok(asyncFn instanceof $AsyncFunction, `${fnName  } instanceof %AsyncFunction%`);
		t.ok($isProto($AsyncFunctionPrototype, asyncFn), `%AsyncFunctionPrototype% is prototype of ${  fnName}`);
	});

	t.end();
});

test('async generator functions', { skip: asyncGenFns.length === 0 }, (t) => {
	const $AsyncGeneratorFunction = GetIntrinsic('%AsyncGeneratorFunction%');
	const $AsyncGeneratorFunctionPrototype = GetIntrinsic('%AsyncGenerator%');
	const $AsyncGeneratorPrototype = GetIntrinsic('%AsyncGeneratorPrototype%');

	forEach(asyncGenFns, (asyncGenFn) => {
		let fnName = asyncGenFn.name;
		fnName = fnName ? `'${  fnName  }'` : 'asyncGenFn';

		t.ok(asyncGenFn instanceof $AsyncGeneratorFunction, `${fnName  } instanceof %AsyncGeneratorFunction%`);
		t.ok($isProto($AsyncGeneratorFunctionPrototype, asyncGenFn), `%AsyncGenerator% is prototype of ${  fnName}`);
		t.ok($isProto($AsyncGeneratorPrototype, asyncGenFn.prototype), `%AsyncGeneratorPrototype% is prototype of ${  fnName  }.prototype`);
	});

	t.end();
});

test('%ThrowTypeError%', (t) => {
	const $ThrowTypeError = GetIntrinsic('%ThrowTypeError%');

	t.equal(typeof $ThrowTypeError, 'function', 'is a function');
	t['throws'](
		$ThrowTypeError,
		TypeError,
		'%ThrowTypeError% throws a TypeError'
	);

	t.end();
});

test('allowMissing', { skip: asyncGenFns.length > 0 }, (t) => {
	t['throws'](
		() => { GetIntrinsic('%AsyncGeneratorPrototype%'); },
		TypeError,
		'throws when missing'
	);

	t.equal(
		GetIntrinsic('%AsyncGeneratorPrototype%', true),
		undefined,
		'does not throw when allowMissing'
	);

	t.end();
});
