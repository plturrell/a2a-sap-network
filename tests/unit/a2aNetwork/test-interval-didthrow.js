'use strict';

const asyncHook = require('../');
const assert = require('assert');

const eventOrder = [];

let throwFlag = null;

asyncHook.addHooks({
  init: function (uid, handle) {
    eventOrder.push(`init#${uid} ${handle.constructor.name}`);
  },
  pre: function (uid) {
    eventOrder.push(`pre#${uid}`);
  },
  post: function (uid, handle, didThrow) {
    throwFlag = didThrow;
    eventOrder.push(`post#${uid}`);
  },
  destroy: function (uid) {
    eventOrder.push(`destroy#${uid}`);
  }
});

asyncHook.enable();

process.once('uncaughtException', () => {
  eventOrder.push('exception');
});

setInterval(() => {
  eventOrder.push('callback');
  throw new Error('error');
});

asyncHook.disable();

process.once('exit', () => {
  assert.strictEqual(throwFlag, true);
  assert.deepEqual(eventOrder, [
    'init#-1 IntervalWrap', 'pre#-1',
    'callback', 'exception',
    'post#-1', 'destroy#-1'
  ]);
});
