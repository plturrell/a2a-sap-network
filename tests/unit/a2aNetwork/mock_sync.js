const path = require('path');
const test = require('tape');
const resolve = require('../');

test('mock', (t) => {
    t.plan(4);

    const files = {};
    files[path.resolve('/foo/bar/baz.js')] = 'beep';

    const dirs = {};
    dirs[path.resolve('/foo/bar')] = true;

    function opts(basedir) {
        return {
            basedir: path.resolve(basedir),
            isFile: function (file) {
                return Object.prototype.hasOwnProperty.call(files, path.resolve(file));
            },
            isDirectory: function (dir) {
                return !!dirs[path.resolve(dir)];
            },
            readFileSync: function (file) {
                return files[path.resolve(file)];
            },
            realpathSync: function (file) {
                return file;
            }
        };
    }

    t.equal(
        resolve.sync('./baz', opts('/foo/bar')),
        path.resolve('/foo/bar/baz.js')
    );

    t.equal(
        resolve.sync('./baz.js', opts('/foo/bar')),
        path.resolve('/foo/bar/baz.js')
    );

    t.throws(() => {
        resolve.sync('baz', opts('/foo/bar'));
    });

    t.throws(() => {
        resolve.sync('../baz', opts('/foo/bar'));
    });
});

test('mock package', (t) => {
    t.plan(1);

    const files = {};
    files[path.resolve('/foo/node_modules/bar/baz.js')] = 'beep';
    files[path.resolve('/foo/node_modules/bar/package.json')] = JSON.stringify({
        main: './baz.js'
    });

    const dirs = {};
    dirs[path.resolve('/foo')] = true;
    dirs[path.resolve('/foo/node_modules')] = true;

    function opts(basedir) {
        return {
            basedir: path.resolve(basedir),
            isFile: function (file) {
                return Object.prototype.hasOwnProperty.call(files, path.resolve(file));
            },
            isDirectory: function (dir) {
                return !!dirs[path.resolve(dir)];
            },
            readFileSync: function (file) {
                return files[path.resolve(file)];
            },
            realpathSync: function (file) {
                return file;
            }
        };
    }

    t.equal(
        resolve.sync('bar', opts('/foo')),
        path.resolve('/foo/node_modules/bar/baz.js')
    );
});

test('symlinked', (t) => {
    t.plan(2);

    const files = {};
    files[path.resolve('/foo/bar/baz.js')] = 'beep';
    files[path.resolve('/foo/bar/symlinked/baz.js')] = 'beep';

    const dirs = {};
    dirs[path.resolve('/foo/bar')] = true;
    dirs[path.resolve('/foo/bar/symlinked')] = true;

    function opts(basedir) {
        return {
            preserveSymlinks: false,
            basedir: path.resolve(basedir),
            isFile: function (file) {
                return Object.prototype.hasOwnProperty.call(files, path.resolve(file));
            },
            isDirectory: function (dir) {
                return !!dirs[path.resolve(dir)];
            },
            readFileSync: function (file) {
                return files[path.resolve(file)];
            },
            realpathSync: function (file) {
                const resolved = path.resolve(file);

                if (resolved.indexOf('symlinked') >= 0) {
                    return resolved;
                }

                const ext = path.extname(resolved);

                if (ext) {
                    const dir = path.dirname(resolved);
                    const base = path.basename(resolved);
                    return path.join(dir, 'symlinked', base);
                }
                return path.join(resolved, 'symlinked');
            }
        };
    }

    t.equal(
        resolve.sync('./baz', opts('/foo/bar')),
        path.resolve('/foo/bar/symlinked/baz.js')
    );

    t.equal(
        resolve.sync('./baz.js', opts('/foo/bar')),
        path.resolve('/foo/bar/symlinked/baz.js')
    );
});

test('readPackageSync', (t) => {
    t.plan(3);

    const files = {};
    files[path.resolve('/foo/node_modules/bar/something-else.js')] = 'beep';
    files[path.resolve('/foo/node_modules/bar/package.json')] = JSON.stringify({
        main: './baz.js'
    });
    files[path.resolve('/foo/node_modules/bar/baz.js')] = 'boop';

    const dirs = {};
    dirs[path.resolve('/foo')] = true;
    dirs[path.resolve('/foo/node_modules')] = true;

    function opts(basedir, useReadPackage) {
        return {
            basedir: path.resolve(basedir),
            isFile: function (file) {
                return Object.prototype.hasOwnProperty.call(files, path.resolve(file));
            },
            isDirectory: function (dir) {
                return !!dirs[path.resolve(dir)];
            },
            readFileSync: useReadPackage ? null : function (file) {
                return files[path.resolve(file)];
            },
            realpathSync: function (file) {
                return file;
            }
        };
    }
    t.test('with readFile', (st) => {
        st.plan(1);

        st.equal(
            resolve.sync('bar', opts('/foo')),
            path.resolve('/foo/node_modules/bar/baz.js')
        );
    });

    const readPackageSync = function (readFileSync, file) {
        if (file.indexOf(path.join('bar', 'package.json')) >= 0) {
            return { main: './something-else.js' };
        }
        return JSON.parse(files[path.resolve(file)]);
    };

    t.test('with readPackage', (st) => {
        st.plan(1);

        const options = opts('/foo');
        delete options.readFileSync;
        options.readPackageSync = readPackageSync;

        st.equal(
            resolve.sync('bar', options),
            path.resolve('/foo/node_modules/bar/something-else.js')
        );
    });

    t.test('with readFile and readPackage', (st) => {
        st.plan(1);

        const options = opts('/foo');
        options.readPackageSync = readPackageSync;
        st.throws(
            () => { resolve.sync('bar', options); },
            TypeError,
            'errors when both readFile and readPackage are provided'
        );
    });
});

