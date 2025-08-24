'use strict';
Object.defineProperty(exports, '__esModule', { value: true });
const chai_1 = require('chai');
const index_1 = require('../index');
/**
 * A function that takes a tracer factory, and tests wheter the initialized tracer
 * fulfills Opentracing's api requirements.
 *
 * @param {object} createTracer - a factory function that allocates a tracer.
 * @param {object} [options] - the options to be set on api compatibility
 */
function apiCompatibilityChecks(createTracer, options) {
    if (createTracer === void 0) { createTracer = function () { return new index_1.Tracer(); }; }
    if (options === void 0) { options = { skipBaggageChecks: false, skipInjectExtractChecks: false }; }
    describe('OpenTracing API Compatibility', () => {
        let tracer;
        let span;
        beforeEach(() => {
            tracer = createTracer();
            span = tracer.startSpan('test-span');
        });
        describe('Tracer', () => {
            describe('startSpan', () => {
                it('should handle Spans and SpanContexts', () => {
                    chai_1.expect(() => { tracer.startSpan('child', { childOf: span }); }).to.not.throw(Error);
                    chai_1.expect(() => { tracer.startSpan('child', { childOf: span.context() }); }).to.not.throw(Error);
                });
            });
            describe('inject', () => {
                (options.skipInjectExtractChecks ? it.skip : it)('should not throw exception on required carrier types', () => {
                    const spanContext = span.context();
                    const textCarrier = {};
                    const binCarrier = new index_1.BinaryCarrier([1, 2, 3]);
                    chai_1.expect(() => { tracer.inject(spanContext, index_1.FORMAT_TEXT_MAP, textCarrier); }).to.not.throw(Error);
                    chai_1.expect(() => { tracer.inject(spanContext, index_1.FORMAT_BINARY, binCarrier); }).to.not.throw(Error);
                    chai_1.expect(() => { tracer.inject(spanContext, index_1.FORMAT_BINARY, {}); }).to.not.throw(Error);
                });
                (options.skipInjectExtractChecks ? it.skip : it)('should handle Spans and SpanContexts', () => {
                    const textCarrier = {};
                    chai_1.expect(() => { tracer.inject(span, index_1.FORMAT_TEXT_MAP, textCarrier); }).to.not.throw(Error);
                    chai_1.expect(() => { tracer.inject(span.context(), index_1.FORMAT_TEXT_MAP, textCarrier); }).to.not.throw(Error);
                });
            });
            describe('extract', () => {
                (options.skipInjectExtractChecks ? it.skip : it)('should not throw exception on required carrier types', () => {
                    const textCarrier = {};
                    const binCarrier = new index_1.BinaryCarrier([1, 2, 3]);
                    chai_1.expect(() => { tracer.extract(index_1.FORMAT_TEXT_MAP, textCarrier); }).to.not.throw(Error);
                    chai_1.expect(() => { tracer.extract(index_1.FORMAT_BINARY, binCarrier); }).to.not.throw(Error);
                    chai_1.expect(() => { tracer.extract(index_1.FORMAT_BINARY, {}); }).to.not.throw(Error);
                    chai_1.expect(() => { tracer.extract(index_1.FORMAT_BINARY, { buffer: null }); }).to.not.throw(Error);
                });
            });
        });
        describe('Span', () => {
            (options.skipBaggageChecks ? it.skip : it)('should set baggage and retrieve baggage', () => {
                span.setBaggageItem('some-key', 'some-value');
                const val = span.getBaggageItem('some-key');
                chai_1.assert.equal('some-value', val);
            });
            describe('finish', () => {
                it('should not throw exceptions on valid arguments', () => {
                    span = tracer.startSpan('test-span');
                    chai_1.expect(() => { return span.finish(Date.now()); }).to.not.throw(Error);
                });
            });
        });
        describe('Reference', () => {
            it('should handle Spans and span.context()', () => {
                chai_1.expect(() => { return new index_1.Reference(index_1.REFERENCE_CHILD_OF, span); }).to.not.throw(Error);
                chai_1.expect(() => { return new index_1.Reference(index_1.REFERENCE_CHILD_OF, span.context()); }).to.not.throw(Error);
            });
        });
        describe('SpanContext', () => {
            describe('toTraceId', () => {
                it('should return a string', () => {
                    span = tracer.startSpan('test-span');
                    console.log(span.context().toTraceId());
                    chai_1.expect(() => { return span.context().toTraceId(); }).to.not.throw(Error);
                    chai_1.expect(span.context().toTraceId()).to.be.a('string');
                });
            });
            describe('toSpanId', () => {
                it('should return a string', () => {
                    span = tracer.startSpan('test-span');
                    chai_1.expect(() => { return span.context().toSpanId(); }).to.not.throw(Error);
                    chai_1.expect(span.context().toSpanId()).to.be.a('string');
                });
            });
        });
    });
}
exports.default = apiCompatibilityChecks;
//# sourceMappingURL=api_compatibility.js.map