'use strict';
const __extends = (this && this.__extends) || (function () {
    let extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (const p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, '__esModule', { value: true });
const chai_1 = require('chai');
const opentracing = require('../index');
function opentracingAPITests() {
    describe('Opentracing API', () => {
        let tracer;
        let span;
        beforeEach(() => {
            tracer = new opentracing.Tracer();
            span = tracer.startSpan('test-span');
        });
        describe('Constants', () => {
            const constStrings = [
                'FORMAT_TEXT_MAP',
                'FORMAT_BINARY',
                'FORMAT_HTTP_HEADERS',
                'REFERENCE_CHILD_OF',
                'REFERENCE_FOLLOWS_FROM'
            ];
            const _loop_1 = function (name_1) {
                it(`${name_1  } should be a constant string`, () => {
                    chai_1.expect(opentracing[name_1]).to.be.a('string');
                });
            };
            for (let _i = 0, constStrings_1 = constStrings; _i < constStrings_1.length; _i++) {
                const name_1 = constStrings_1[_i];
                _loop_1(name_1);
            }
        });
        describe('Standalone functions', () => {
            const funcs = [
                'childOf',
                'followsFrom',
                'initGlobalTracer',
                'globalTracer'
            ];
            const _loop_2 = function (name_2) {
                it(`${name_2  } should be a function`, () => {
                    chai_1.expect(opentracing[name_2]).to.be.a('function');
                });
            };
            for (let _i = 0, funcs_1 = funcs; _i < funcs_1.length; _i++) {
                const name_2 = funcs_1[_i];
                _loop_2(name_2);
            }
            describe('global tracer', () => {
                const dummySpan = new opentracing.Span();
                afterEach(() => {
                    opentracing.initGlobalTracer(new opentracing.Tracer());
                });
                it('should use the global tracer', () => {
                    opentracing.initGlobalTracer(new TestTracer());
                    const tracer = opentracing.globalTracer();
                    const span = tracer.startSpan('test');
                    chai_1.expect(span).to.equal(dummySpan);
                });
                var TestTracer = /** @class */ (function (_super) {
                    __extends(TestTracer, _super);
                    function TestTracer() {
                        return _super !== null && _super.apply(this, arguments) || this;
                    }
                    TestTracer.prototype._startSpan = function (name, fields) {
                        return dummySpan;
                    };
                    return TestTracer;
                }(opentracing.Tracer));
            });
        });
        describe('Tracer', () => {
            it('should be a class', () => {
                chai_1.expect(new opentracing.Tracer()).to.be.an('object');
            });
        });
        describe('Span', () => {
            it('should be a class', () => {
                chai_1.expect(span).to.be.an('object');
            });
        });
        describe('SpanContext', () => {
            it('should be a class', () => {
                const spanContext = span.context();
                chai_1.expect(spanContext).to.be.an('object');
            });
        });
        describe('Reference', () => {
            it('should be a class', () => {
                const ref = new opentracing.Reference(opentracing.REFERENCE_CHILD_OF, span.context());
                chai_1.expect(ref).to.be.an('object');
            });
        });
        describe('BinaryCarrier', () => {
            it('should set binary data as a field called "buffer"', () => {
                const buffer = new Float64Array(10);
                const ref = new opentracing.BinaryCarrier(buffer);
                chai_1.expect(ref.buffer).to.equal(buffer);
            });
        });
    });
}
exports.opentracingAPITests = opentracingAPITests;
exports.default = opentracingAPITests;
//# sourceMappingURL=opentracing_api.js.map