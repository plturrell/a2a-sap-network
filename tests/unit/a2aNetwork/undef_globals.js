const test = require('tape');
const vm = require('vm');
const fs = require('fs');
const src = fs.readFileSync(`${__dirname  }/../../index.js`, 'utf8');

test('u8a without globals', (t) => {
    const c = {
        module: { exports: {} },
    };
    c.exports = c.module.exports;
    vm.runInNewContext(src, c);
    const TA = c.module.exports;
    const ua = new(TA.Uint8Array)(5);
    
    t.equal(ua.length, 5);
    ua[1] = 256 + 55;
    t.equal(ua[1], 55);
    t.end();
});
