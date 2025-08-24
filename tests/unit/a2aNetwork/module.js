const find = require('../');
const test = require('tap').test;

test('single file', (t) => {
    t.plan(2);
    
    const finder = find(__filename);
    const files = [];
    finder.on('file', (file) => {
        t.equal(file, __filename);
        files.push(file);
    });
    
    finder.on('directory', (dir) => {
        t.fail(dir);
    });
    
    finder.on('end', () => {
        t.deepEqual(files, [ __filename ]);
    });
});
