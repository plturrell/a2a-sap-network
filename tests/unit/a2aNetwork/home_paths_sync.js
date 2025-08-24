'use strict';

const fs = require('fs');
const homedir = require('../lib/homedir');
const path = require('path');

const test = require('tape');
const mkdirp = require('mkdirp');
const rimraf = require('rimraf');
const mv = require('mv');
const copyDir = require('copy-dir');
const tmp = require('tmp');

const HOME = homedir();

const hnm = path.join(HOME, '.node_modules');
const hnl = path.join(HOME, '.node_libraries');

const resolve = require('../sync');

function makeDir(t, dir, cb) {
    mkdirp(dir, (err) => {
        if (err) {
            cb(err);
        } else {
            t.teardown(() => {
                rimraf.sync(dir);
            });
            cb();
        }
    });
}

function makeTempDir(t, dir, cb) {
    if (fs.existsSync(dir)) {
        const tmpResult = tmp.dirSync();
        t.teardown(tmpResult.removeCallback);
        const backup = path.join(tmpResult.name, path.basename(dir));
        mv(dir, backup, (err) => {
            if (err) {
                cb(err);
            } else {
                t.teardown(() => {
                    mv(backup, dir, cb);
                });
                makeDir(t, dir, cb);
            }
        });
    } else {
        makeDir(t, dir, cb);
    }
}

test('homedir module paths', (t) => {
    t.plan(7);

    makeTempDir(t, hnm, (err) => {
        t.error(err, 'no error with HNM temp dir');
        if (err) {
            return t.end();
        }

        const bazHNMDir = path.join(hnm, 'baz');
        const dotMainDir = path.join(hnm, 'dot_main');
        copyDir.sync(path.join(__dirname, 'resolver/baz'), bazHNMDir);
        copyDir.sync(path.join(__dirname, 'resolver/dot_main'), dotMainDir);

        const bazHNMmain = path.join(bazHNMDir, 'quux.js');
        t.equal(require.resolve('baz'), bazHNMmain, 'sanity check: require.resolve finds HNM `baz`');
        const dotMainMain = path.join(dotMainDir, 'index.js');
        t.equal(require.resolve('dot_main'), dotMainMain, 'sanity check: require.resolve finds `dot_main`');

        makeTempDir(t, hnl, (err) => {
            t.error(err, 'no error with HNL temp dir');
            if (err) {
                return t.end();
            }
            const bazHNLDir = path.join(hnl, 'baz');
            copyDir.sync(path.join(__dirname, 'resolver/baz'), bazHNLDir);

            const dotSlashMainDir = path.join(hnl, 'dot_slash_main');
            const dotSlashMainMain = path.join(dotSlashMainDir, 'index.js');
            copyDir.sync(path.join(__dirname, 'resolver/dot_slash_main'), dotSlashMainDir);

            t.equal(require.resolve('baz'), bazHNMmain, 'sanity check: require.resolve finds HNM `baz`');
            t.equal(require.resolve('dot_slash_main'), dotSlashMainMain, 'sanity check: require.resolve finds HNL `dot_slash_main`');

            t.test('with temp dirs', (st) => {
                st.plan(3);

                st.test('just in `$HOME/.node_modules`', (s2t) => {
                    s2t.plan(1);

                    const res = resolve('dot_main');
                    s2t.equal(res, dotMainMain, '`dot_main` resolves in `$HOME/.node_modules`');
                });

                st.test('just in `$HOME/.node_libraries`', (s2t) => {
                    s2t.plan(1);

                    const res = resolve('dot_slash_main');
                    s2t.equal(res, dotSlashMainMain, '`dot_slash_main` resolves in `$HOME/.node_libraries`');
                });

                st.test('in `$HOME/.node_libraries` and `$HOME/.node_modules`', (s2t) => {
                    s2t.plan(1);

                    const res = resolve('baz');
                    s2t.equal(res, bazHNMmain, '`baz` resolves in `$HOME/.node_modules` when in both');
                });
            });
        });
    });
});
