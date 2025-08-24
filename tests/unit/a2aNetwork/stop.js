const find = require('../');
const test = require('tap').test;
const path = require('path');

test('stop', (t) => {
    t.plan(1);
    
    const finder = find(`${__dirname  }/..`);
    const files = [];
    let stopped = false;
    finder.on('file', (file) => {
        files.push(file);
        if (files.length === 3) {
            finder.stop();
            stopped = true;
        }
        else if (stopped) {
            t.fail('files didn\'t stop');
        }
    });
    
    finder.on('directory', (dir, stat, stop) => {
        if (stopped) t.fail('directories didn\'t stop');
    });
    
    finder.on('end', () => {
        t.fail('shouldn\'t have ended');
    });
    
    finder.on('stop', () => {
        t.equal(files.length, 3);
    });
});
