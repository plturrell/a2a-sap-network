const test = require('tape');
const path = require('path');
const parse = path.parse || require('path-parse');
const keys = require('object-keys');

const nodeModulesPaths = require('../lib/node-modules-paths');

const verifyDirs = function verifyDirs(t, start, dirs, moduleDirectories, paths) {
    const moduleDirs = [].concat(moduleDirectories || 'node_modules');
    if (paths) {
        for (let k = 0; k < paths.length; ++k) {
            moduleDirs.push(path.basename(paths[k]));
        }
    }

    const foundModuleDirs = {};
    const uniqueDirs = {};
    const parsedDirs = {};
    for (let i = 0; i < dirs.length; ++i) {
        const parsed = parse(dirs[i]);
        if (!foundModuleDirs[parsed.base]) { foundModuleDirs[parsed.base] = 0; }
        foundModuleDirs[parsed.base] += 1;
        parsedDirs[parsed.dir] = true;
        uniqueDirs[dirs[i]] = true;
    }
    t.equal(keys(parsedDirs).length >= start.split(path.sep).length, true, 'there are >= dirs than "start" has');
    const foundModuleDirNames = keys(foundModuleDirs);
    t.deepEqual(foundModuleDirNames, moduleDirs, 'all desired module dirs were found');
    t.equal(keys(uniqueDirs).length, dirs.length, 'all dirs provided were unique');

    const counts = {};
    for (let j = 0; j < foundModuleDirNames.length; ++j) {
        counts[foundModuleDirs[j]] = true;
    }
    t.equal(keys(counts).length, 1, 'all found module directories had the same count');
};

test('node-modules-paths', (t) => {
    t.test('no options', (t) => {
        const start = path.join(__dirname, 'resolver');
        const dirs = nodeModulesPaths(start);

        verifyDirs(t, start, dirs);

        t.end();
    });

    t.test('empty options', (t) => {
        const start = path.join(__dirname, 'resolver');
        const dirs = nodeModulesPaths(start, {});

        verifyDirs(t, start, dirs);

        t.end();
    });

    t.test('with paths=array option', (t) => {
        const start = path.join(__dirname, 'resolver');
        const paths = ['a', 'b'];
        const dirs = nodeModulesPaths(start, { paths: paths });

        verifyDirs(t, start, dirs, null, paths);

        t.end();
    });

    t.test('with paths=function option', (t) => {
        const paths = function paths(request, absoluteStart, getNodeModulesDirs, opts) {
            return getNodeModulesDirs().concat(path.join(absoluteStart, 'not node modules', request));
        };

        const start = path.join(__dirname, 'resolver');
        const dirs = nodeModulesPaths(start, { paths: paths }, 'pkg');

        verifyDirs(t, start, dirs, null, [path.join(start, 'not node modules', 'pkg')]);

        t.end();
    });

    t.test('with paths=function skipping node modules resolution', (t) => {
        const paths = function paths(request, absoluteStart, getNodeModulesDirs, opts) {
            return [];
        };
        const start = path.join(__dirname, 'resolver');
        const dirs = nodeModulesPaths(start, { paths: paths });
        t.deepEqual(dirs, [], 'no node_modules was computed');
        t.end();
    });

    t.test('with moduleDirectory option', (t) => {
        const start = path.join(__dirname, 'resolver');
        const moduleDirectory = 'not node modules';
        const dirs = nodeModulesPaths(start, { moduleDirectory: moduleDirectory });

        verifyDirs(t, start, dirs, moduleDirectory);

        t.end();
    });

    t.test('with 1 moduleDirectory and paths options', (t) => {
        const start = path.join(__dirname, 'resolver');
        const paths = ['a', 'b'];
        const moduleDirectory = 'not node modules';
        const dirs = nodeModulesPaths(start, { paths: paths, moduleDirectory: moduleDirectory });

        verifyDirs(t, start, dirs, moduleDirectory, paths);

        t.end();
    });

    t.test('with 1+ moduleDirectory and paths options', (t) => {
        const start = path.join(__dirname, 'resolver');
        const paths = ['a', 'b'];
        const moduleDirectories = ['not node modules', 'other modules'];
        const dirs = nodeModulesPaths(start, { paths: paths, moduleDirectory: moduleDirectories });

        verifyDirs(t, start, dirs, moduleDirectories, paths);

        t.end();
    });

    t.test('combine paths correctly on Windows', (t) => {
        const start = 'C:\\Users\\username\\myProject\\src';
        const paths = [];
        const moduleDirectories = ['node_modules', start];
        const dirs = nodeModulesPaths(start, { paths: paths, moduleDirectory: moduleDirectories });

        t.equal(dirs.indexOf(path.resolve(start)) > -1, true, 'should contain start dir');

        t.end();
    });

    t.test('combine paths correctly on non-Windows', { skip: process.platform === 'win32' }, (t) => {
        const start = '/Users/username/git/myProject/src';
        const paths = [];
        const moduleDirectories = ['node_modules', '/Users/username/git/myProject/src'];
        const dirs = nodeModulesPaths(start, { paths: paths, moduleDirectory: moduleDirectories });

        t.equal(dirs.indexOf(path.resolve(start)) > -1, true, 'should contain start dir');

        t.end();
    });
});
