'use strict';

const test = require('tape');

const setDunderProto = require('../set');

test('setDunderProto', { skip: !setDunderProto }, (t) => {
	if (!setDunderProto) {
		throw 'should never happen; this is just for type narrowing'; // eslint-disable-line no-throw-literal
	}

	// @ts-expect-error
	t['throws'](() => { setDunderProto(); }, TypeError, 'throws if no arguments');
	// @ts-expect-error
	t['throws'](() => { setDunderProto(undefined); }, TypeError, 'throws with undefined and nothing');
	// @ts-expect-error
	t['throws'](() => { setDunderProto(undefined, undefined); }, TypeError, 'throws with undefined and undefined');
	// @ts-expect-error
	t['throws'](() => { setDunderProto(null); }, TypeError, 'throws with null and undefined');
	// @ts-expect-error
	t['throws'](() => { setDunderProto(null, undefined); }, TypeError, 'throws with null and undefined');

	/** @type {{ inherited?: boolean }} */
	const obj = {};
	t.ok('toString' in obj, 'object initially has toString');

	setDunderProto(obj, null);
	t.notOk('toString' in obj, 'object no longer has toString');

	t.notOk('inherited' in obj, 'object lacks inherited property');
	setDunderProto(obj, { inherited: true });
	t.equal(obj.inherited, true, 'object has inherited property');

	t.end();
});

test('no dunder proto', { skip: !!setDunderProto }, (t) => {
	if ('__proto__' in Object.prototype) {
		t['throws'](
			// @ts-expect-error
			() => { ({}).__proto__ = null; }, // eslint-disable-line no-proto
			Error,
			'throws when setting Object.prototype.__proto__'
		);
	} else {
		t.notOk('__proto__' in Object.prototype, 'no __proto__ in Object.prototype');
	}

	t.end();
});
