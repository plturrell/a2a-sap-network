const path = require('path');
const fs = require('fs');
const test = require('tape');
const resolve = require('../');
const async = require('../async');

test('`./async` entry point', (t) => {
    t.equal(resolve, async, '`./async` entry point is the same as `main`');
    t.end();
});

test('async foo', (t) => {
    t.plan(12);
    const dir = path.join(__dirname, 'resolver');

    resolve('./foo', { basedir: dir }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'foo.js'));
        t.equal(pkg && pkg.name, 'resolve');
    });

    resolve('./foo.js', { basedir: dir }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'foo.js'));
        t.equal(pkg && pkg.name, 'resolve');
    });

    resolve('./foo', { basedir: dir, 'package': { main: 'resolver' } }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'foo.js'));
        t.equal(pkg && pkg.main, 'resolver');
    });

    resolve('./foo.js', { basedir: dir, 'package': { main: 'resolver' } }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'foo.js'));
        t.equal(pkg.main, 'resolver');
    });

    resolve('./foo', { basedir: dir, filename: path.join(dir, 'baz.js') }, (err, res) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'foo.js'));
    });

    resolve('foo', { basedir: dir }, (err) => {
        t.equal(err.message, `Cannot find module 'foo' from '${  path.resolve(dir)  }'`);
        t.equal(err.code, 'MODULE_NOT_FOUND');
    });

    // Test that filename is reported as the "from" value when passed.
    resolve('foo', { basedir: dir, filename: path.join(dir, 'baz.js') }, (err) => {
        t.equal(err.message, `Cannot find module 'foo' from '${  path.join(dir, 'baz.js')  }'`);
    });
});

test('bar', (t) => {
    t.plan(6);
    const dir = path.join(__dirname, 'resolver');

    resolve('foo', { basedir: `${dir  }/bar` }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'bar/node_modules/foo/index.js'));
        t.equal(pkg, undefined);
    });

    resolve('foo', { basedir: `${dir  }/bar` }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'bar/node_modules/foo/index.js'));
        t.equal(pkg, undefined);
    });

    resolve('foo', { basedir: `${dir  }/bar`, 'package': { main: 'bar' } }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'bar/node_modules/foo/index.js'));
        t.equal(pkg.main, 'bar');
    });
});

test('baz', (t) => {
    t.plan(4);
    const dir = path.join(__dirname, 'resolver');

    resolve('./baz', { basedir: dir }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'baz/quux.js'));
        t.equal(pkg.main, 'quux.js');
    });

    resolve('./baz', { basedir: dir, 'package': { main: 'resolver' } }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'baz/quux.js'));
        t.equal(pkg.main, 'quux.js');
    });
});

test('biz', (t) => {
    t.plan(24);
    const dir = path.join(__dirname, 'resolver/biz/node_modules');

    resolve('./grux', { basedir: dir }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'grux/index.js'));
        t.equal(pkg, undefined);
    });

    resolve('./grux', { basedir: dir, 'package': { main: 'biz' } }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'grux/index.js'));
        t.equal(pkg.main, 'biz');
    });

    resolve('./garply', { basedir: dir }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'garply/lib/index.js'));
        t.equal(pkg.main, './lib');
    });

    resolve('./garply', { basedir: dir, 'package': { main: 'biz' } }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'garply/lib/index.js'));
        t.equal(pkg.main, './lib');
    });

    resolve('tiv', { basedir: `${dir  }/grux` }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'tiv/index.js'));
        t.equal(pkg, undefined);
    });

    resolve('tiv', { basedir: `${dir  }/grux`, 'package': { main: 'grux' } }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'tiv/index.js'));
        t.equal(pkg.main, 'grux');
    });

    resolve('tiv', { basedir: `${dir  }/garply` }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'tiv/index.js'));
        t.equal(pkg, undefined);
    });

    resolve('tiv', { basedir: `${dir  }/garply`, 'package': { main: './lib' } }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'tiv/index.js'));
        t.equal(pkg.main, './lib');
    });

    resolve('grux', { basedir: `${dir  }/tiv` }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'grux/index.js'));
        t.equal(pkg, undefined);
    });

    resolve('grux', { basedir: `${dir  }/tiv`, 'package': { main: 'tiv' } }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'grux/index.js'));
        t.equal(pkg.main, 'tiv');
    });

    resolve('garply', { basedir: `${dir  }/tiv` }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'garply/lib/index.js'));
        t.equal(pkg.main, './lib');
    });

    resolve('garply', { basedir: `${dir  }/tiv`, 'package': { main: 'tiv' } }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'garply/lib/index.js'));
        t.equal(pkg.main, './lib');
    });
});

test('quux', (t) => {
    t.plan(2);
    const dir = path.join(__dirname, 'resolver/quux');

    resolve('./foo', { basedir: dir, 'package': { main: 'quux' } }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'foo/index.js'));
        t.equal(pkg.main, 'quux');
    });
});

test('normalize', (t) => {
    t.plan(2);
    const dir = path.join(__dirname, 'resolver/biz/node_modules/grux');

    resolve('../grux', { basedir: dir }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'index.js'));
        t.equal(pkg, undefined);
    });
});

test('cup', (t) => {
    t.plan(5);
    const dir = path.join(__dirname, 'resolver');

    resolve('./cup', { basedir: dir, extensions: ['.js', '.coffee'] }, (err, res) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'cup.coffee'));
    });

    resolve('./cup.coffee', { basedir: dir }, (err, res) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'cup.coffee'));
    });

    resolve('./cup', { basedir: dir, extensions: ['.js'] }, (err, res) => {
        t.equal(err.message, `Cannot find module './cup' from '${  path.resolve(dir)  }'`);
        t.equal(err.code, 'MODULE_NOT_FOUND');
    });

    // Test that filename is reported as the "from" value when passed.
    resolve('./cup', { basedir: dir, extensions: ['.js'], filename: path.join(dir, 'cupboard.js') }, (err, res) => {
        t.equal(err.message, `Cannot find module './cup' from '${  path.join(dir, 'cupboard.js')  }'`);
    });
});

test('mug', (t) => {
    t.plan(3);
    const dir = path.join(__dirname, 'resolver');

    resolve('./mug', { basedir: dir }, (err, res) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'mug.js'));
    });

    resolve('./mug', { basedir: dir, extensions: ['.coffee', '.js'] }, (err, res) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, '/mug.coffee'));
    });

    resolve('./mug', { basedir: dir, extensions: ['.js', '.coffee'] }, (err, res) => {
        t.equal(res, path.join(dir, '/mug.js'));
    });
});

test('other path', (t) => {
    t.plan(6);
    const resolverDir = path.join(__dirname, 'resolver');
    const dir = path.join(resolverDir, 'bar');
    const otherDir = path.join(resolverDir, 'other_path');

    resolve('root', { basedir: dir, paths: [otherDir] }, (err, res) => {
        if (err) t.fail(err);
        t.equal(res, path.join(resolverDir, 'other_path/root.js'));
    });

    resolve('lib/other-lib', { basedir: dir, paths: [otherDir] }, (err, res) => {
        if (err) t.fail(err);
        t.equal(res, path.join(resolverDir, 'other_path/lib/other-lib.js'));
    });

    resolve('root', { basedir: dir }, (err, res) => {
        t.equal(err.message, `Cannot find module 'root' from '${  path.resolve(dir)  }'`);
        t.equal(err.code, 'MODULE_NOT_FOUND');
    });

    resolve('zzz', { basedir: dir, paths: [otherDir] }, (err, res) => {
        t.equal(err.message, `Cannot find module 'zzz' from '${  path.resolve(dir)  }'`);
        t.equal(err.code, 'MODULE_NOT_FOUND');
    });
});

test('path iterator', (t) => {
    t.plan(2);

    const resolverDir = path.join(__dirname, 'resolver');

    const exactIterator = function (x, start, getPackageCandidates, opts) {
        return [path.join(resolverDir, x)];
    };

    resolve('baz', { packageIterator: exactIterator }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(resolverDir, 'baz/quux.js'));
        t.equal(pkg && pkg.name, 'baz');
    });
});

test('incorrect main', (t) => {
    t.plan(1);

    const resolverDir = path.join(__dirname, 'resolver');
    const dir = path.join(resolverDir, 'incorrect_main');

    resolve('./incorrect_main', { basedir: resolverDir }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'index.js'));
    });
});

test('missing index', (t) => {
    t.plan(2);

    const resolverDir = path.join(__dirname, 'resolver');
    resolve('./missing_index', { basedir: resolverDir }, (err, res, pkg) => {
        t.ok(err instanceof Error);
        t.equal(err && err.code, 'MODULE_NOT_FOUND', 'error has correct error code');
    });
});

test('missing main', (t) => {
    t.plan(1);

    const resolverDir = path.join(__dirname, 'resolver');

    resolve('./missing_main', { basedir: resolverDir }, (err, res, pkg) => {
        t.equal(err && err.code, 'MODULE_NOT_FOUND', 'error has correct error code');
    });
});

test('null main', (t) => {
    t.plan(1);

    const resolverDir = path.join(__dirname, 'resolver');

    resolve('./null_main', { basedir: resolverDir }, (err, res, pkg) => {
        t.equal(err && err.code, 'MODULE_NOT_FOUND', 'error has correct error code');
    });
});

test('main: false', (t) => {
    t.plan(2);

    const basedir = path.join(__dirname, 'resolver');
    const dir = path.join(basedir, 'false_main');
    resolve('./false_main', { basedir: basedir }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(
            res,
            path.join(dir, 'index.js'),
            '`"main": false`: resolves to `index.js`'
        );
        t.deepEqual(pkg, {
            name: 'false_main',
            main: false
        });
    });
});

test('without basedir', (t) => {
    t.plan(1);

    const dir = path.join(__dirname, 'resolver/without_basedir');
    const tester = require(path.join(dir, 'main.js')); // eslint-disable-line global-require

    tester(t, (err, res, pkg) => {
        if (err) {
            t.fail(err);
        } else {
            t.equal(res, path.join(dir, 'node_modules/mymodule.js'));
        }
    });
});

test('#52 - incorrectly resolves module-paths like "./someFolder/" when there is a file of the same name', (t) => {
    t.plan(2);

    const dir = path.join(__dirname, 'resolver');

    resolve('./foo', { basedir: path.join(dir, 'same_names') }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'same_names/foo.js'));
    });

    resolve('./foo/', { basedir: path.join(dir, 'same_names') }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'same_names/foo/index.js'));
    });
});

test('#211 - incorrectly resolves module-paths like "." when from inside a folder with a sibling file of the same name', (t) => {
    t.plan(2);

    const dir = path.join(__dirname, 'resolver');

    resolve('./', { basedir: path.join(dir, 'same_names/foo') }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'same_names/foo/index.js'));
    });

    resolve('.', { basedir: path.join(dir, 'same_names/foo') }, (err, res, pkg) => {
        if (err) t.fail(err);
        t.equal(res, path.join(dir, 'same_names/foo/index.js'));
    });
});

test('async: #121 - treating an existing file as a dir when no basedir', (t) => {
    const testFile = path.basename(__filename);

    t.test('sanity check', (st) => {
        st.plan(1);
        resolve(`./${  testFile}`, (err, res, pkg) => {
            if (err) t.fail(err);
            st.equal(res, __filename, 'sanity check');
        });
    });

    t.test('with a fake directory', (st) => {
        st.plan(4);

        resolve(`./${  testFile  }/blah`, (err, res, pkg) => {
            st.ok(err, 'there is an error');
            st.notOk(res, 'no result');

            st.equal(err && err.code, 'MODULE_NOT_FOUND', 'error code matches require.resolve');
            st.equal(
                err && err.message,
                `Cannot find module './${  testFile  }/blah' from '${  __dirname  }'`,
                'can not find nonexistent module'
            );
            st.end();
        });
    });

    t.end();
});

test('async dot main', (t) => {
    const start = new Date();
    t.plan(3);
    resolve('./resolver/dot_main', (err, ret) => {
        t.notOk(err);
        t.equal(ret, path.join(__dirname, 'resolver/dot_main/index.js'));
        t.ok(new Date() - start < 50, 'resolve.sync timedout');
        t.end();
    });
});

test('async dot slash main', (t) => {
    const start = new Date();
    t.plan(3);
    resolve('./resolver/dot_slash_main', (err, ret) => {
        t.notOk(err);
        t.equal(ret, path.join(__dirname, 'resolver/dot_slash_main/index.js'));
        t.ok(new Date() - start < 50, 'resolve.sync timedout');
        t.end();
    });
});

test('not a directory', (t) => {
    t.plan(6);
    const path = './foo';
    resolve(path, { basedir: __filename }, function (err, res, pkg) {
        t.ok(err, 'a non-directory errors');
        t.equal(arguments.length, 1);
        t.equal(res, undefined);
        t.equal(pkg, undefined);

        t.equal(err && err.message, `Cannot find module '${  path  }' from '${  __filename  }'`);
        t.equal(err && err.code, 'MODULE_NOT_FOUND');
    });
});

test('non-string "main" field in package.json', (t) => {
    t.plan(5);

    const dir = path.join(__dirname, 'resolver');
    resolve('./invalid_main', { basedir: dir }, (err, res, pkg) => {
        t.ok(err, 'errors on non-string main');
        t.equal(err.message, 'package “invalid_main” `main` must be a string');
        t.equal(err.code, 'INVALID_PACKAGE_MAIN');
        t.equal(res, undefined, 'res is undefined');
        t.equal(pkg, undefined, 'pkg is undefined');
    });
});

test('non-string "main" field in package.json', (t) => {
    t.plan(5);

    const dir = path.join(__dirname, 'resolver');
    resolve('./invalid_main', { basedir: dir }, (err, res, pkg) => {
        t.ok(err, 'errors on non-string main');
        t.equal(err.message, 'package “invalid_main” `main` must be a string');
        t.equal(err.code, 'INVALID_PACKAGE_MAIN');
        t.equal(res, undefined, 'res is undefined');
        t.equal(pkg, undefined, 'pkg is undefined');
    });
});

test('browser field in package.json', (t) => {
    t.plan(3);

    const dir = path.join(__dirname, 'resolver');
    resolve(
        './browser_field',
        {
            basedir: dir,
            packageFilter: function packageFilter(pkg) {
                if (pkg.browser) {
                    pkg.main = pkg.browser; // eslint-disable-line no-param-reassign
                    delete pkg.browser; // eslint-disable-line no-param-reassign
                }
                return pkg;
            }
        },
        (err, res, pkg) => {
            if (err) t.fail(err);
            t.equal(res, path.join(dir, 'browser_field', 'b.js'));
            t.equal(pkg && pkg.main, 'b');
            t.equal(pkg && pkg.browser, undefined);
        }
    );
});

test('absolute paths', (t) => {
    t.plan(4);

    const extensionless = __filename.slice(0, -path.extname(__filename).length);

    resolve(__filename, (err, res) => {
        t.equal(
            res,
            __filename,
            'absolute path to this file resolves'
        );
    });
    resolve(extensionless, (err, res) => {
        t.equal(
            res,
            __filename,
            'extensionless absolute path to this file resolves'
        );
    });
    resolve(__filename, { basedir: process.cwd() }, (err, res) => {
        t.equal(
            res,
            __filename,
            'absolute path to this file with a basedir resolves'
        );
    });
    resolve(extensionless, { basedir: process.cwd() }, (err, res) => {
        t.equal(
            res,
            __filename,
            'extensionless absolute path to this file with a basedir resolves'
        );
    });
});

const malformedDir = path.join(__dirname, 'resolver/malformed_package_json');
test('malformed package.json', { skip: !fs.existsSync(malformedDir) }, (t) => {
    /* eslint operator-linebreak: ["error", "before"], function-paren-newline: "off" */
    t.plan(
        (3 * 3) // 3 sets of 3 assertions in the final callback
        + 2 // 1 readPackage call with malformed package.json
    );

    const basedir = malformedDir;
    const expected = path.join(basedir, 'index.js');

    resolve('./index.js', { basedir: basedir }, (err, res, pkg) => {
        t.error(err, 'no error');
        t.equal(res, expected, 'malformed package.json is silently ignored');
        t.equal(pkg, undefined, 'malformed package.json gives an undefined `pkg` argument');
    });

    resolve(
        './index.js',
        {
            basedir: basedir,
            packageFilter: function (pkg, pkgfile, dir) {
                t.fail('should not reach here');
            }
        },
        (err, res, pkg) => {
            t.error(err, 'with packageFilter: no error');
            t.equal(res, expected, 'with packageFilter: malformed package.json is silently ignored');
            t.equal(pkg, undefined, 'with packageFilter: malformed package.json gives an undefined `pkg` argument');
        }
    );

    resolve(
        './index.js',
        {
            basedir: basedir,
            readPackage: function (readFile, pkgfile, cb) {
                t.equal(pkgfile, path.join(basedir, 'package.json'), 'readPackageSync: `pkgfile` is package.json path');
                readFile(pkgfile, (err, result) => {
                    try {
                        cb(null, JSON.parse(result));
                    } catch (e) {
                        t.ok(e instanceof SyntaxError, 'readPackage: malformed package.json parses as a syntax error');
                        cb(null);
                    }
                });
            }
        },
        (err, res, pkg) => {
            t.error(err, 'with readPackage: no error');
            t.equal(res, expected, 'with readPackage: malformed package.json is silently ignored');
            t.equal(pkg, undefined, 'with readPackage: malformed package.json gives an undefined `pkg` argument');
        }
    );
});
