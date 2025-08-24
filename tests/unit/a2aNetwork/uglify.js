
const test = require('tap').test;
const uglify = require('uglify-js');
const path = require('path');

test('can be uglified', (t) => {
  const files = ['format.js', 'index.js', 'stack-chain.js'].map((filename) => {
    return path.resolve(__dirname, `../../${  filename}`);
  });
  uglify.minify(files);
  t.end();
});
