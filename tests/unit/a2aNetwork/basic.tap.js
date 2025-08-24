'use strict';

const test            = require('tap').test;
const wrapEmitter     = require('../listener.js');
const Emitter         = require('events').EventEmitter;
const ServerResponse  = require('http').ServerResponse;
const IncomingMessage = require('http').IncomingMessage;

test('bindEmitter', (t) => {
  t.plan(9);

  t.test('with no parameters', (t) => {
    t.plan(1);

    t.throws(
      () => { wrapEmitter(); },
      new Error('can only wrap real EEs'),
      'validates that it has an EE'
    );
  });

  t.test('with only an emitter', (t) => {
    t.plan(1);

    t.throws(
      () => { wrapEmitter(new Emitter()); },
      new Error('must have function to run on listener addition'),
      'requires a marking function'
    );
  });

  t.test('with only an emitter and a marker', (t) => {
    t.plan(1);

    t.throws(
      () => { wrapEmitter(new Emitter(), () => {}); },
      new Error('must have function to wrap listeners when emitting'),
      'requires a preparation function'
    );
  });

  t.test('with all required parameters', (t) => {
    t.plan(5);

    function nop() {}
    function passthrough(value) { return value; }

    const ee = new Emitter();
    const numPropsBeforeWrap = Object.keys(ee).length;

    t.doesNotThrow(
      () => { wrapEmitter(ee, nop, passthrough); },
      'monkeypatches correctly'
    );

    t.ok(ee.__wrapped, 'is marked as being a wrapped emitter');

    ee.on('test', (value) => {
      t.equal(value, 8, 'value was still passed through');
    });

    t.doesNotThrow(() => { ee.emit('test', 8); }, 'emitting still works');

    const numPropsAfterWrap = Object.keys(ee).length;
    t.equal(numPropsAfterWrap, numPropsBeforeWrap,
      'doesn\'t add extra enumerable properties');
  });

  t.test('when a listener removes another listener', (t) => {
    t.plan(4);

    const ee = new Emitter();
    function listener1() { /* nop */ }
    function listener2() { ee.removeListener('listen', listener2); }

    function nop() {}
    function wrap(handler) {
      return function () {
        return handler.apply(this, arguments);
      };
    }
    wrapEmitter(ee, nop, wrap);

    ee.on('listen', listener1);
    ee.on('listen', listener2);
    t.equal(ee.listeners('listen').length, 2, 'both listeners are there');

    t.doesNotThrow(() => {
      ee.emit('listen');
    }, 'emitting still works');
    t.equal(ee.listeners('listen').length, 1, 'one listener got removed');
    t.equal(ee.listeners('listen')[0], listener1, 'the right listener is still there');
  });

  t.test('when listener explodes', (t) => {
    t.plan(4);

    const ee = new Emitter();
    wrapEmitter(
      ee,
      () => {},
      (handler) => {
        return function wrapped() {
          handler.apply(this, arguments);
        };
      }
    );

    function kaboom() {
      throw new Error('whoops');
    }

    ee.on('bad', kaboom);

    t.throws(() => { ee.emit('bad'); });
    t.equal(typeof ee.removeListener, 'function', 'removeListener is still there');
    t.notOk(ee.removeListener.__wrapped, 'removeListener got unwrapped');
    t.equal(ee._events.bad, kaboom, 'listener isn\'t still bound');
  });

  t.test('when unwrapping emitter', (t) => {
    t.plan(9);

    const ee = new Emitter();
    wrapEmitter(
      ee,
      () => {},
      (handler) => { return handler; }
    );

    t.ok(ee.addListener.__wrapped, 'addListener is wrapped');
    t.ok(ee.on.__wrapped, 'on is wrapped');
    t.ok(ee.emit.__wrapped, 'emit is wrapped');
    t.notOk(ee.removeListener.__wrapped, 'removeListener is not wrapped');

    t.doesNotThrow(() => { ee.__unwrap(); }, 'can unwrap without dying');

    t.notOk(ee.addListener.__wrapped, 'addListener is unwrapped');
    t.notOk(ee.on.__wrapped, 'on is unwrapped');
    t.notOk(ee.emit.__wrapped, 'emit is unwrapped');
    t.notOk(ee.removeListener.__wrapped, 'removeListener is unwrapped');
  });

  t.test('when wrapping the same emitter multiple times', (t) => {
    t.plan(6);

    const ee = new Emitter();
    const values = [];
    wrapEmitter(
      ee,
      () => { values.push(1); },
      (handler) => { return handler; }
    );

    wrapEmitter(
      ee,
      () => { values.push(2); },
      (handler) => { return handler; }
    );

    ee.on('test', (value) => {
      t.equal(value, 31, 'got expected value');
      t.deepEqual(values, [1, 2], 'both marker functions were called');
    });

    t.ok(ee.addListener.__wrapped, 'addListener is wrapped');
    t.ok(ee.on.__wrapped, 'on is wrapped');
    t.ok(ee.emit.__wrapped, 'emit is wrapped');
    t.notOk(ee.removeListener.__wrapped, 'removeListener is not wrapped');

    ee.emit('test', 31);
  });

  t.test('when adding multiple handlers to a ServerResponse', (t) => {
    t.plan(1);

    const ee = new ServerResponse(new IncomingMessage());
    const values = [];

    ee.on('test', (_) => {});
    ee.on('test', (_) => {});

    wrapEmitter(
      ee,
      () => { values.push(1); },
      (handler) => { return handler; }
    );

    ee.on('test', (_) => {});

    t.deepEqual(values, [1], 'marker function was not called');
  });
});
