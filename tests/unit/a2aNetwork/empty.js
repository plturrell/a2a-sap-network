const mkdirp = require('mkdirp');
const test = require('tap').test;
const find = require('../');

mkdirp.sync(`${__dirname  }/empty`);

test('empty', (t) => {
    t.plan(1);
    const w = find(`${__dirname  }/empty`);
    const files = [];
    w.on('file', (file) => {
        files.push(file);
    });
    w.on('end', () => {
        t.deepEqual(files, []);
    });
});
