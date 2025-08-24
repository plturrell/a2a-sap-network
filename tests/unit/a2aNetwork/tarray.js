const TA = require('../');
const test = require('tape');

test('tiny u8a test', (t) => {
    const ua = new(TA.Uint8Array)(5);
    t.equal(ua.length, 5);
    ua[1] = 256 + 55;
    t.equal(ua[1], 55);
    t.end();
});
