/*
 * Jake JavaScript build tool
 * Copyright 2112 Matthew Eernisse (mde@fleegix.org)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
*/

const PROJECT_DIR = process.env.PROJECT_DIR;
const JAKE_CMD = `${PROJECT_DIR}/bin/cli.js`;

const assert = require('assert');
const fs = require('fs');
const exec = require('child_process').execSync;
const { rmRf } = require(`${PROJECT_DIR}/lib/jake`);

const cleanUpAndNext = function (callback) {
  rmRf('./foo', {
    silent: true
  });
  callback && callback();
};

suite('fileTask', function () {
  this.timeout(7000);

  setup(() => {
    cleanUpAndNext();
  });

  test('where a file-task prereq does not change with --always-make', () => {
    let out;
    out = exec(`${JAKE_CMD} -q fileTest:foo/from-src1.txt`).toString().trim();
    assert.equal('fileTest:foo/src1.txt task\nfileTest:foo/from-src1.txt task',
      out);
    out = exec(`${JAKE_CMD} -q -B fileTest:foo/from-src1.txt`).toString().trim();
    assert.equal('fileTest:foo/src1.txt task\nfileTest:foo/from-src1.txt task',
      out);
    cleanUpAndNext();
  });

  test('concating two files', () => {
    let out;
    out = exec(`${JAKE_CMD} -q fileTest:foo/concat.txt`).toString().trim();
    assert.equal('fileTest:foo/src1.txt task\ndefault task\nfileTest:foo/src2.txt task\n' +
          'fileTest:foo/concat.txt task', out);
    // Check to see the two files got concat'd
    const data = fs.readFileSync(`${process.cwd()  }/foo/concat.txt`);
    assert.equal('src1src2', data.toString());
    cleanUpAndNext();
  });

  test('where a file-task prereq does not change', () => {
    let out;
    out = exec(`${JAKE_CMD} -q fileTest:foo/from-src1.txt`).toString().trim();
    assert.equal('fileTest:foo/src1.txt task\nfileTest:foo/from-src1.txt task', out);
    out = exec(`${JAKE_CMD} -q fileTest:foo/from-src1.txt`).toString().trim();
    // Second time should be a no-op
    assert.equal('', out);
    cleanUpAndNext();
  });

  test('where a file-task prereq does change, then does not', (next) => {
    exec('mkdir -p ./foo');
    exec('touch ./foo/from-src1.txt');
    setTimeout(() => {
      fs.writeFileSync('./foo/src1.txt', '-SRC');
      // Task should run the first time
      let out;
      out = exec(`${JAKE_CMD} -q fileTest:foo/from-src1.txt`).toString().trim();
      assert.equal('fileTest:foo/from-src1.txt task', out);
      // Task should not run on subsequent invocation
      out = exec(`${JAKE_CMD} -q fileTest:foo/from-src1.txt`).toString().trim();
      assert.equal('', out);
      cleanUpAndNext(next);
    }, 1000);
  });

  test('a preexisting file', () => {
    const prereqData = 'howdy';
    exec('mkdir -p ./foo');
    fs.writeFileSync('foo/prereq.txt', prereqData);
    let out;
    out = exec(`${JAKE_CMD} -q fileTest:foo/from-prereq.txt`).toString().trim();
    assert.equal('fileTest:foo/from-prereq.txt task', out);
    const data = fs.readFileSync(`${process.cwd()  }/foo/from-prereq.txt`);
    assert.equal(prereqData, data.toString());
    out = exec(`${JAKE_CMD} -q fileTest:foo/from-prereq.txt`).toString().trim();
    // Second time should be a no-op
    assert.equal('', out);
    cleanUpAndNext();
  });

  test('a preexisting file with --always-make flag', () => {
    const prereqData = 'howdy';
    exec('mkdir -p ./foo');
    fs.writeFileSync('foo/prereq.txt', prereqData);
    let out;
    out = exec(`${JAKE_CMD} -q fileTest:foo/from-prereq.txt`).toString().trim();
    assert.equal('fileTest:foo/from-prereq.txt task', out);
    const data = fs.readFileSync(`${process.cwd()  }/foo/from-prereq.txt`);
    assert.equal(prereqData, data.toString());
    out = exec(`${JAKE_CMD} -q -B fileTest:foo/from-prereq.txt`).toString().trim();
    assert.equal('fileTest:foo/from-prereq.txt task', out);
    cleanUpAndNext();
  });

  test('nested directory-task', () => {
    exec(`${JAKE_CMD} -q fileTest:foo/bar/baz/bamf.txt`);
    const data = fs.readFileSync(`${process.cwd()  }/foo/bar/baz/bamf.txt`);
    assert.equal('w00t', data);
    cleanUpAndNext();
  });

  test('partially existing prereqs', () => {
    /*
     dependency graph:
                               /-- foo/output2a.txt --\
     foo -- foo/output1.txt --+                        +-- output3.txt
                               \-- foo/output2b.txt --/
    */
    // build part of the prereqs
    exec(`${JAKE_CMD} -q fileTest:foo/output2a.txt`);
    // verify the final target gets built
    exec(`${JAKE_CMD} -q fileTest:foo/output3.txt`);
    const data = fs.readFileSync(`${process.cwd()  }/foo/output3.txt`);
    assert.equal('w00t', data);
    cleanUpAndNext();
  });
});

