const fs = require('fs');

const loader = require('./loader');

const NUM_TIMES = 10;
const BASE_DIR = `${__dirname  }/perf`;
const TREES = ['../test/arrtree', 'rbtree', 'bintree'];

function mean(arr) {
    let sum = 0;
    arr.forEach((n) => {
        sum += n;
    });
    return sum/arr.length;
}

function timeit(f) {
    const diffs = [];
    for(let i=0; i < NUM_TIMES; i++) {
        const start = Date.now();
        f();
        const end = Date.now();

        const diff = (end - start)/1000;
        diffs.push(diff);
    }
    return diffs;
}

function print_times(arr) {
    console.log('Mean: ', mean(arr));
}

function build(tree_class, test_path) {
    const tests = loader.load(test_path);
    const inserts = loader.get_inserts(tests);

    console.log('build tree...');
    print_times(timeit(() =>{
        loader.build_tree(tree_class, inserts);
    }));
}

function build_destroy(tree_class, test_path) {
    const tests = loader.load(test_path);
    const inserts = loader.get_inserts(tests);
    const removes = loader.get_removes(tests);

    console.log('build/destroy tree...');
    print_times(timeit(() => {
        const tree = loader.build_tree(tree_class, inserts);
        removes.forEach((n) => {
            tree.remove(n);
        });
    }));
}

function find(tree_class, test_path) {
    const tests = loader.load(test_path);
    const inserts = loader.get_inserts(tests);

    const tree = loader.build_tree(tree_class, inserts);
    console.log('find all nodes...');
    print_times(timeit(() => {
        inserts.forEach((n) => {
            tree.find(n);
        });
    }));
}


function interleaved(tree_class, test_path) {
    const tests = loader.load(test_path);

    console.log('interleaved build/destroy...');
    print_times(timeit(() => {
        const tree = new tree_class((a,b) => { return a - b; });
        tests.forEach((n) => {
            if(n > 0)
                tree.insert(n);
            else
                tree.remove(n);
        });
    }));
}

const tests = fs.readdirSync(BASE_DIR);

const test_funcs = {};
TREES.forEach((tree) => {
    const tree_class = require(`../lib/${  tree}`);
    tests.forEach((test) => {
       const test_path = `${BASE_DIR  }/${  test}`;
       test_funcs[`${tree  }_${  test  }_build`] = function(assert) {
          build(tree_class, test_path);
          assert.done();
       };
       test_funcs[`${tree  }_${  test  }_build_destroy`] = function(assert) {
          build_destroy(tree_class, test_path);
          assert.done();
       };
       test_funcs[`${tree  }_${  test  }_find`] = function(assert) {
          find(tree_class, test_path);
          assert.done();
       };
       test_funcs[`${tree  }_${  test  }_interleaved`] = function(assert) {
          interleaved(tree_class, test_path);
          assert.done();
       };
    });
});

exports.performance = test_funcs;
