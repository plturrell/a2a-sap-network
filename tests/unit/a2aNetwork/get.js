'use strict';

const test = require('tape');

const getDunderProto = require('../get');

test('getDunderProto', { skip: !getDunderProto }, (t) => {
	if (!getDunderProto) {
		throw 'should never happen; this is just for type narrowing'; // eslint-disable-line no-throw-literal
	}

	// @ts-expect-error
	t['throws'](() => { getDunderProto(); }, TypeError, 'throws if no argument');
	// @ts-expect-error
	t['throws'](() => { getDunderProto(undefined); }, TypeError, 'throws with undefined');
	// @ts-expect-error
	t['throws'](() => { getDunderProto(null); }, TypeError, 'throws with null');

	t.equal(getDunderProto({}), Object.prototype);
	t.equal(getDunderProto([]), Array.prototype);
	t.equal(getDunderProto(() => {}), Function.prototype);
	t.equal(getDunderProto(/./g), RegExp.prototype);
	t.equal(getDunderProto(42), Number.prototype);
	t.equal(getDunderProto(true), Boolean.prototype);
	t.equal(getDunderProto('foo'), String.prototype);

	t.end();
});

test('no dunder proto', { skip: !!getDunderProto }, (t) => {
	t.notOk('__proto__' in Object.prototype, 'no __proto__ in Object.prototype');

	t.end();
});
