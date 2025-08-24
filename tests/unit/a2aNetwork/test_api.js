const _ = require('underscore');

const loader = require('./loader');

const SAMPLE_FILE = `${__dirname  }/samples/10k`;
const TREES = ['rbtree', 'bintree'];

function clear(assert, tree_class) {
    const inserts = loader.get_inserts(loader.load(SAMPLE_FILE));
    const tree = loader.build_tree(tree_class, inserts);
    tree.clear();
    inserts.forEach((data) => {
        assert.equal(tree.find(data), null);
    });
}

function dup(assert, tree_class) {
    const tree = loader.new_tree(tree_class);

    assert.ok(tree.insert(100));
    assert.ok(tree.insert(101));
    assert.ok(!tree.insert(101));
    assert.ok(!tree.insert(100));
    tree.remove(100);
    assert.ok(!tree.insert(101));
    assert.ok(tree.insert(100));
    assert.ok(!tree.insert(100));
}

function nonexist(assert, tree_class) {
    const tree = loader.new_tree(tree_class);

    assert.ok(!tree.remove(100));
    tree.insert(100);
    assert.ok(!tree.remove(101));
    assert.ok(tree.remove(100));
}

function minmax(assert, tree_class) {
    let tree = loader.new_tree(tree_class);
    assert.equal(tree.min(), null);
    assert.equal(tree.max(), null);

    const inserts = loader.get_inserts(loader.load(SAMPLE_FILE));
    tree = loader.build_tree(tree_class, inserts);

    assert.equal(tree.min(), _.min(inserts));
    assert.equal(tree.max(), _.max(inserts));
}

function forward_it(assert, tree_class) {
    const inserts = loader.get_inserts(loader.load(SAMPLE_FILE));
    const tree = loader.build_tree(tree_class, inserts);

    let items = [];
    let it=tree.iterator(), data;
    while((data = it.next()) !== null) {
        items.push(data);
    }

    inserts.sort((a,b) => { return a - b; });

    assert.deepEqual(items, inserts);

    items = [];
    tree.each((data) => {
        items.push(data);
    });

    assert.deepEqual(items, inserts);
}

function forward_it_break(assert, tree_class) {
    const inserts = loader.get_inserts(loader.load(SAMPLE_FILE));
    const tree = loader.build_tree(tree_class, inserts);

    let items = [];
    let it=tree.iterator(), data;
    while((data = it.next()) !== null) {
        items.push(data);
    }

    inserts.sort((a,b) => { return a - b; });

    assert.deepEqual(items, inserts);

    items = [];
    let i = 0;
    tree.each((data) => {
        items.push(data);
        if (i === 3) {
            return false;
        }
        i++;
    });

    assert.equal(items.length, 4);
}

function reverse_it(assert, tree_class) {
    const inserts = loader.get_inserts(loader.load(SAMPLE_FILE));
    const tree = loader.build_tree(tree_class, inserts);

    let items = [];

    let it=tree.iterator(), data;
    while((data = it.prev()) !== null) {
        items.push(data);
    }

    inserts.sort((a,b) => { return b - a; });

    assert.deepEqual(items, inserts);

    items = [];
    tree.reach((data) => {
        items.push(data);
    });

    assert.deepEqual(items, inserts);
}

function reverse_it_break(assert, tree_class) {
    const inserts = loader.get_inserts(loader.load(SAMPLE_FILE));
    const tree = loader.build_tree(tree_class, inserts);

    let items = [];

    let it=tree.iterator(), data;
    while((data = it.prev()) !== null) {
        items.push(data);
    }

    inserts.sort((a,b) => { return b - a; });

    assert.deepEqual(items, inserts);

    items = [];
    let i = 0;
    tree.reach((data) => {
        items.push(data);
        if (i === 3) {
            return false;
        }
        i++;
    });

    assert.equal(items.length, 4);
}

function switch_it(assert, tree_class) {
    const inserts = loader.get_inserts(loader.load(SAMPLE_FILE));
    const tree = loader.build_tree(tree_class, inserts);

    inserts.sort((a,b) => { return a - b; });

    function do_switch(after) {
        const items = [];
        const it = tree.iterator();
        for(let i = 0; i < after; i++) {
            items.push(it.next());
        }

        while((data = it.prev()) !== null) {
            items.push(data);
        }

        const forward = inserts.slice(0, after);
        const reverse = inserts.slice(0, after - 1).reverse();
        const all = forward.concat(reverse);

        assert.deepEqual(items, all);
    }

    do_switch(1);
    do_switch(10);
    do_switch(1000);
    do_switch(9000);
}

function empty_it(assert, tree_class) {
    const tree = loader.new_tree(tree_class);

    let it = tree.iterator();
    assert.equal(it.next(), null);

    it = tree.iterator();
    assert.equal(it.prev(), null);
}

function lower_bound(assert, tree_class) {
    const inserts = loader.get_inserts(loader.load(SAMPLE_FILE));
    const tree = loader.build_tree(tree_class, inserts);

    inserts.sort((a,b) => { return a - b; });

    for(let i=1; i<inserts.length-1; ++i) {
        const item = inserts[i];

        var iter = tree.lowerBound(item);
        assert.equal(iter.data(), item);
        assert.equal(iter.prev(), inserts[i-1]);
        iter.next();
        assert.equal(iter.next(), inserts[i+1]);

        const prev = tree.lowerBound(item - 0.1);
        assert.equal(prev.data(), inserts[i]);

        const next = tree.lowerBound(item + 0.1);
        assert.equal(next.data(), inserts[i+1]);
    }

    // test edges
    var iter = tree.lowerBound(-1);
    assert.equal(iter.data(), inserts[0]);
    const last = inserts[inserts.length - 1];
    iter = tree.lowerBound(last);
    assert.equal(iter.data(), last);
    iter = tree.lowerBound(last + 1);
    assert.equal(iter.data(), null);
}

function upper_bound(assert, tree_class) {
    const inserts = loader.get_inserts(loader.load(SAMPLE_FILE));
    const tree = loader.build_tree(tree_class, inserts);

    inserts.sort((a,b) => { return a - b; });

    for(let i=0; i<inserts.length-2; ++i) {
        const item = inserts[i];

        var iter = tree.upperBound(item);
        assert.equal(iter.data(), inserts[i+1]);
        assert.equal(iter.prev(), inserts[i]);
        iter.next();
        assert.equal(iter.next(), inserts[i+2]);

        const prev = tree.upperBound(item - 0.1);
        assert.equal(prev.data(), inserts[i]);

        const next = tree.upperBound(item + 0.1);
        assert.equal(next.data(), inserts[i+1]);
    }

    // test edges
    var iter = tree.upperBound(-1);
    assert.equal(iter.data(), inserts[0]);
    const last = inserts[inserts.length - 1];
    iter = tree.upperBound(last);
    assert.equal(iter.data(), null);
    iter = tree.upperBound(last + 1);
    assert.equal(iter.data(), null);

    // test empty
    const empty = new tree_class((a,b) => { return a.val - b.val; });
    var iter = empty.upperBound({val:0});
    assert.equal(iter.data(), null);
}

function find(assert, tree_class) {
    const inserts = loader.get_inserts(loader.load(SAMPLE_FILE));
    const tree = loader.build_tree(tree_class, inserts);

    for(let i=1; i<inserts.length-1; ++i) {
        const item = inserts[i];

        assert.equal(tree.find(item), item);
        assert.equal(tree.find(item + 0.1), null);
    }
}

function find_iter(assert, tree_class) {
    const inserts = loader.get_inserts(loader.load(SAMPLE_FILE));
    const tree = loader.build_tree(tree_class, inserts);

    inserts.sort((a,b) => { return a - b; });

    for(let i=1; i<inserts.length-1; ++i) {
        const item = inserts[i];

        const iter = tree.findIter(item);
        assert.equal(iter.data(), item);
        assert.equal(iter.prev(), inserts[i-1]);
        iter.next();
        assert.equal(iter.next(), inserts[i+1]);

        assert.equal(tree.findIter(item + 0.1), null);
    }
}


const TESTS = {
    clear: clear,
    dup: dup,
    nonexist: nonexist,
    minmax: minmax,
    forward_it: forward_it,
    forward_it_break: forward_it_break,
    reverse_it: reverse_it,
    reverse_it_break: reverse_it_break,
    switch_it: switch_it,
    empty_it: empty_it,
    lower_bound: lower_bound,
    upper_bound: upper_bound,
    find: find,
    find_iter: find_iter
};

const test_funcs = {};
TREES.forEach((tree) => {
    const tree_class = require(`../lib/${  tree}`);
    for(const test in TESTS) {
        (function(test) {
            test_funcs[`${tree  }_${  test}`] = function(assert) {
                TESTS[test](assert, tree_class);
                assert.done();
            };
        })(test);
    }
});

exports.api = test_funcs;
