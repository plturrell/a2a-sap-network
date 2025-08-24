
const test = require('tap').test;
const chain = require('../../');
const produce = require('../produce.js');

Error.stackTraceLimit = Infinity;

test('stack extend part', (t) => {
  const modify = function (text) {
    return function (error, frames) {
      if (error.test) {
        frames.push(text);
      }

      return frames;
    };
  };

  t.test('no extend modifier attached', (t) => {
    const error = new Error();
        error.test = error;

    const original = error.callSite.original.length;
    const mutated = error.callSite.mutated.length;
    t.strictEqual(mutated, original);

    t.end();
  });

  t.test('attach modifier', (t) => {
    const error = new Error();
        error.test = error;

    const wonderLand = modify('wonder land');

    chain.extend.attach(wonderLand);

    const original = error.callSite.original.length;
    const mutated = error.callSite.mutated.length;
    t.strictEqual(mutated, original + 1);

    chain.extend.deattach(wonderLand);

    t.end();
  });

  t.test('setting callSite', (t) => {
    const error = new Error();
        error.test = error;
        error.correct = true;

    error.callSite = 'custom';
    t.strictEqual(error.callSite, 'custom');
    error.stack;
    t.strictEqual(error.callSite, 'custom');

    t.end();
  });

  t.end();
});
