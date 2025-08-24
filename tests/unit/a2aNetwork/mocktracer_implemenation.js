'use strict';
Object.defineProperty(exports, '__esModule', { value: true });
const chai_1 = require('chai');
const index_1 = require('../index');
function mockTracerimplementationTests() {
    describe('Mock Tracer API tests', () => {
        describe('Tracer#report', () => {
            it('should not throw exceptions when running report', () => {
                const tracer = new index_1.MockTracer();
                const span = tracer.startSpan('test_operation');
                span.addTags({ key: 'value' });
                span.finish();
                chai_1.expect(() => {
                    const report = tracer.report();
                    for (let _i = 0, _a = report.spans; _i < _a.length; _i++) {
                        const span_1 = _a[_i];
                        span_1.tags();
                    }
                }).to.not.throw(Error);
            });
        });
    });
}
exports.default = mockTracerimplementationTests;
//# sourceMappingURL=mocktracer_implemenation.js.map