'use strict';

// stdlib
const tap = require('tap');
const test = tap.test;
const EventEmitter = require('events').EventEmitter;

// module under test
const context = require('../context.js');

// multiple contexts in use
const tracer = context.createNamespace('tracer');

function Trace(harvester) {
  this.harvester = harvester;
}

Trace.prototype.runHandler = function (handler) {
  const trace = tracer.run(handler);
  this.harvester.emit('finished', trace.transaction);
};


test('simple tracer built on contexts', (t) => {
  t.plan(6);

  const harvester = new EventEmitter();
  const trace = new Trace(harvester);

  harvester.on('finished', (transaction) => {
    t.ok(transaction, 'transaction should have been passed in');
    t.equal(transaction.status, 'ok', 'transaction should have finished OK');
    t.equal(Object.keys(process.namespaces).length, 1, 'Should only have one namespace.');
  });

  trace.runHandler(() => {
    t.ok(tracer.active, 'tracer should have an active context');
    tracer.set('transaction', {status : 'ok'});
    t.ok(tracer.get('transaction'), 'can retrieve newly-set value');
    t.equal(tracer.get('transaction').status, 'ok', 'value should be correct');
  });
});
