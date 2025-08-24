const test = require('tap').test;
const chain = require('../../');

test('non extensible Error objects don\'t throw', (t) => {
  const error = new Error('don\'t extend me');
  Object.preventExtensions(error);
  t.doesNotThrow(() => {
    error.stack;
  });
  t.end();
});

test('stack is correct on non extensible error object', (t) => {
  const error = new Error('don\'t extend me');
  Object.preventExtensions(error);

  chain.format.replace(() => {
    return 'good';
  });

  try {
    t.equal(error.stack, 'good');
  } catch (e) { t.ifError(e); }

  chain.format.restore();

  t.end();
});

