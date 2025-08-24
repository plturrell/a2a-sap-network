'use strict';
Object.defineProperty(exports, '__esModule', { value: true });
const chai_1 = require('chai');
const index_1 = require('../index');
function noopImplementationTests(createTracer) {
    if (createTracer === void 0) { createTracer = function () { return new index_1.Tracer(); }; }
    describe('Noop Tracer Implementation', () => {
        describe('Tracer#inject', () => {
            it('should handle Spans and SpanContexts', () => {
                const tracer = createTracer();
                const span = tracer.startSpan('test_operation');
                const textCarrier = {};
                chai_1.expect(() => { tracer.inject(span, index_1.FORMAT_TEXT_MAP, textCarrier); }).to.not.throw(Error);
            });
        });
        describe('Span#finish', () => {
            it('should return undefined', () => {
                const tracer = createTracer();
                const span = tracer.startSpan('test_span');
                chai_1.expect(span.finish()).to.be.undefined;
            });
        });
        describe('Miscellaneous', () => {
            describe('Memory usage', () => {
                it('should not report leaks after setting the global tracer', () => {
                    index_1.initGlobalTracer(createTracer());
                });
            });
        });
    });
}
exports.noopImplementationTests = noopImplementationTests;
exports.default = noopImplementationTests;
//# sourceMappingURL=noop_implementation.js.map