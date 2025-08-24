const test = require('tape');
const wordwrap = require('../');

const fs = require('fs');
const idleness = fs.readFileSync(`${__dirname  }/idleness.txt`, 'utf8');

test('stop80', (t) => {
    const lines = wordwrap(80)(idleness).split(/\n/);
    const words = idleness.split(/\s+/);
    
    lines.forEach((line) => {
        t.ok(line.length <= 80, 'line > 80 columns');
        const chunks = line.match(/\S/) ? line.split(/\s+/) : [];
        t.deepEqual(chunks, words.splice(0, chunks.length));
    });
    t.end();
});

test('start20stop60', (t) => {
    const lines = wordwrap(20, 100)(idleness).split(/\n/);
    const words = idleness.split(/\s+/);
    
    lines.forEach((line) => {
        t.ok(line.length <= 100, 'line > 100 columns');
        const chunks = line
            .split(/\s+/)
            .filter((x) => { return x.match(/\S/); })
        ;
        t.deepEqual(chunks, words.splice(0, chunks.length));
        t.deepEqual(line.slice(0, 20), new Array(20 + 1).join(' '));
    });
    t.end();
});
