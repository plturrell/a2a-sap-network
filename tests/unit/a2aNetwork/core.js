const test = require('tape');
const keys = require('object-keys');
const semver = require('semver');

const resolve = require('../');

const brokenNode = semver.satisfies(process.version, '11.11 - 11.13');

test('core modules', (t) => {
    t.test('isCore()', (st) => {
        st.ok(resolve.isCore('fs'));
        st.ok(resolve.isCore('net'));
        st.ok(resolve.isCore('http'));

        st.ok(!resolve.isCore('seq'));
        st.ok(!resolve.isCore('../'));

        st.ok(!resolve.isCore('toString'));

        st.end();
    });

    t.test('core list', (st) => {
        const cores = keys(resolve.core);
        st.plan(cores.length);

        for (let i = 0; i < cores.length; ++i) {
            var mod = cores[i];
            // note: this must be require, not require.resolve, due to https://github.com/nodejs/node/issues/43274
            const requireFunc = function () { require(mod); }; // eslint-disable-line no-loop-func
            t.comment(`${mod  }: ${  resolve.core[mod]}`);
            if (resolve.core[mod]) {
                st.doesNotThrow(requireFunc, `${mod  } supported; requiring does not throw`);
            } else if (brokenNode) {
                st.ok(true, 'this version of node is broken: attempting to require things that fail to resolve breaks "home_paths" tests');
            } else {
                st.throws(requireFunc, `${mod  } not supported; requiring throws`);
            }
        }

        st.end();
    });

    t.test('core via repl module', { skip: !resolve.core.repl }, (st) => {
        const libs = require('repl')._builtinLibs; // eslint-disable-line no-underscore-dangle
        if (!libs) {
            st.skip('module.builtinModules does not exist');
            return st.end();
        }
        for (let i = 0; i < libs.length; ++i) {
            var mod = libs[i];
            st.ok(resolve.core[mod], `${mod  } is a core module`);
            st.doesNotThrow(
                () => { require(mod); }, // eslint-disable-line no-loop-func
                `requiring ${  mod  } does not throw`
            );
        }
        st.end();
    });

    t.test('core via builtinModules list', { skip: !resolve.core.module }, (st) => {
        const libs = require('module').builtinModules;
        if (!libs) {
            st.skip('module.builtinModules does not exist');
            return st.end();
        }
        const blacklist = [
            '_debug_agent',
            'v8/tools/tickprocessor-driver',
            'v8/tools/SourceMap',
            'v8/tools/tickprocessor',
            'v8/tools/profile'
        ];
        for (let i = 0; i < libs.length; ++i) {
            var mod = libs[i];
            if (blacklist.indexOf(mod) === -1) {
                st.ok(resolve.core[mod], `${mod  } is a core module`);
                st.doesNotThrow(
                    () => { require(mod); }, // eslint-disable-line no-loop-func
                    `requiring ${  mod  } does not throw`
                );
            }
        }
        st.end();
    });

    t.end();
});
