const test = require('tap').test;

test('asyncListener preserves function length', (t) => {
  t.plan(2);

  const fsLengthsPre = computeValueLengths(require('fs'));
  const httpLengthsPre = computeValueLengths(require('http'));

  if (!process.addAsyncListener) require('../index.js');

  const fsLengthsPost = computeValueLengths(require('fs'));
  const httpLengthsPost = computeValueLengths(require('http'));

  t.same(fsLengthsPre, fsLengthsPost);
  t.same(httpLengthsPre, httpLengthsPost);
});

function computeValueLengths(o) {
  const lengths = [];
  for (const k in o) {
    lengths.push(o[k].length);
  }
  return lengths;
}
