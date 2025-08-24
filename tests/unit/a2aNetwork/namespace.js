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

// Load the jake global
require(`${PROJECT_DIR}/lib/jake`);
const { Namespace } = require(`${PROJECT_DIR}/lib/namespace`);

require('./jakefile');

const assert = require('assert');

suite('namespace', function () {

  this.timeout(7000);

  test('resolve namespace by relative name', () => {
    let aaa, bbb, ccc;
    aaa = namespace('aaa', () => {
      bbb = namespace('bbb', () => {
        ccc = namespace('ccc', () => {
        });
      });
    });

    assert.ok(aaa, Namespace.ROOT_NAMESPACE.resolveNamespace('aaa'));
    assert.ok(bbb === aaa.resolveNamespace('bbb'));
    assert.ok(ccc === aaa.resolveNamespace('bbb:ccc'));
  });

  test('resolve task in sub-namespace by relative path', () => {
    const curr = Namespace.ROOT_NAMESPACE.resolveNamespace('zooby');
    const task = curr.resolveTask('frang:w00t:bar');
    assert.ok(task.action.toString().indexOf('zooby:frang:w00t:bar') > -1);
  });

  test('prefer local to top-level', () => {
    const curr = Namespace.ROOT_NAMESPACE.resolveNamespace('zooby:frang:w00t');
    const task = curr.resolveTask('bar');
    assert.ok(task.action.toString().indexOf('zooby:frang:w00t:bar') > -1);
  });

  test('does resolve top-level', () => {
    const curr = Namespace.ROOT_NAMESPACE.resolveNamespace('zooby:frang:w00t');
    const task = curr.resolveTask('foo');
    assert.ok(task.action.toString().indexOf('top-level foo') > -1);
  });

  test('absolute lookup works from sub-namespaces', () => {
    const curr = Namespace.ROOT_NAMESPACE.resolveNamespace('hurr:durr');
    const task = curr.resolveTask('zooby:frang:w00t:bar');
    assert.ok(task.action.toString().indexOf('zooby:frang:w00t:bar') > -1);
  });

  test('resolution miss with throw error', () => {
    const curr = Namespace.ROOT_NAMESPACE;
    const task = curr.resolveTask('asdf:qwer');
    assert.ok(!task);
  });

});
