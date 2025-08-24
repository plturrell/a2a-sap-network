'use strict';

const tap = require('tap');
const test = tap.test;
const sinon = require('sinon');
const shimmer = require('../index.js');

test('shimmer initialization', (t) => {
  t.plan(4);

  t.doesNotThrow(() => { shimmer(); });

  const mock = sinon.expectation
    .create('logger')
    .withArgs('no original function undefined to wrap')
    .once();

  t.doesNotThrow(() => {
    shimmer({ logger: mock });
  }, 'initializer doesn\'t throw');

  t.doesNotThrow(() => {
    shimmer.wrap();
  }, 'invoking the wrap method with no params doesn\'t throw');

  t.doesNotThrow(() => {
    mock.verify();
  }, 'logger method was called with the expected message');
});

test('shimmer initialized with non-function logger', (t) => {
  t.plan(2);

  const mock = sinon.expectation
    .create('logger')
    .withArgs('new logger isn\'t a function, not replacing')
    .once();

  shimmer({ logger: mock });

  t.doesNotThrow(() => {
    shimmer({ logger: { ham: 'chunx' } });
  }, 'even bad initialization doesn\'t throw');

  t.doesNotThrow(() => {
    mock.verify();
  }, 'logger initialization failed in the expected way');
});
