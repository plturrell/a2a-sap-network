const test = require('tape');
const through = require('../');

// must emit end before close.

test('end before close', (assert) => {
  const ts = through();
  ts.autoDestroy = false;
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
  assert.notOk(closed);
  ts.destroy();
  assert.ok(closed);
  assert.end();
});

