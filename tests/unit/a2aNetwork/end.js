const test = require('tape');
const through = require('../');

// must emit end before close.

test('end before close', (assert) => {
  const ts = through();
  let ended = false, closed = false;

  ts.on('end', () => {
    assert.ok(!closed);
    ended = true;
  });
  ts.on('close', () => {
    assert.ok(ended);
    closed = true;
  });

  ts.write(1);
  ts.write(2);
  ts.write(3);
  ts.end();
  assert.ok(ended);
  assert.ok(closed);
  assert.end();
});

test('end only once', (t) => {

  const ts = through();
  let ended = false, closed = false;

  ts.on('end', () => {
    t.equal(ended, false);
    ended = true;
  });

  ts.queue(null);
  ts.queue(null);
  ts.queue(null);

  ts.resume();

  t.end();
});
