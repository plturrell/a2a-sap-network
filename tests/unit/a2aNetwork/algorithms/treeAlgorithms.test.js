const { expect } = require('chai');
const sinon = require('sinon');
const TreeAlgorithms = require('../../../srv/algorithms/treeAlgorithms');

/**
 * Unit tests for Tree Algorithms implementation
 * Following SAP testing standards and enterprise patterns
 */
describe('TreeAlgorithms', () => {
    let treeAlgorithms;
    let sandbox;

    beforeEach(() => {
        sandbox = sinon.createSandbox();
        treeAlgorithms = new TreeAlgorithms();
    });

    afterEach(() => {
        sandbox.restore();
    });

    describe('Leaf Detection', () => {
        it('should correctly identify leaf values', () => {
            expect(treeAlgorithms.isLeaf(null)).to.be.true;
            expect(treeAlgorithms.isLeaf(undefined)).to.be.true;
            expect(treeAlgorithms.isLeaf(42)).to.be.true;
            expect(treeAlgorithms.isLeaf('string')).to.be.true;
            expect(treeAlgorithms.isLeaf(true)).to.be.true;
        });

        it('should correctly identify non-leaf containers', () => {
            expect(treeAlgorithms.isLeaf([1, 2, 3])).to.be.false;
            expect(treeAlgorithms.isLeaf({ a: 1 })).to.be.false;
            expect(treeAlgorithms.isLeaf(new Map([['a', 1]]))).to.be.false;
            expect(treeAlgorithms.isLeaf(new Set([1, 2]))).to.be.false;
        });

        it('should identify empty containers as leaves', () => {
            expect(treeAlgorithms.isLeaf([])).to.be.true;
            expect(treeAlgorithms.isLeaf({})).to.be.true;
            expect(treeAlgorithms.isLeaf(new Map())).to.be.true;
            expect(treeAlgorithms.isLeaf(new Set())).to.be.true;
        });
    });

    describe('Flatten Operation', () => {
        it('should flatten simple nested arrays', () => {
            const input = [[1], [[[2, 3]]], [4]];
            const result = treeAlgorithms.flatten(input);
            
            expect(result).to.deep.equal([1, 2, 3, 4]);
        });

        it('should flatten mixed nested structures', () => {
            const input = {
                numbers: [1, [2, 3]],
                nested: {
                    deep: [[4, 5]]
                }
            };
            const result = treeAlgorithms.flatten(input);
            
            expect(result).to.include.members([1, 2, 3, 4, 5]);
        });

        it('should handle Maps and Sets', () => {
            const input = new Map([
                ['array', [1, 2]],
                ['set', new Set([3, 4])]
            ]);
            const result = treeAlgorithms.flatten(input);
            
            expect(result).to.include.members([1, 2, 3, 4]);
        });

        it('should filter out null and undefined values', () => {
            const input = [1, null, [2, undefined, 3]];
            const result = treeAlgorithms.flatten(input);
            
            expect(result).to.deep.equal([1, 2, 3]);
        });

        it('should handle empty structures', () => {
            expect(treeAlgorithms.flatten([])).to.deep.equal([]);
            expect(treeAlgorithms.flatten({})).to.deep.equal([]);
            expect(treeAlgorithms.flatten(null)).to.deep.equal([]);
        });
    });

    describe('Map Structure Operation', () => {
        it('should map function over nested arrays preserving structure', () => {
            const input = [[1], [[[2, 3]]], [4]];
            const result = treeAlgorithms.mapStructure(x => x * 2, input);
            
            expect(result).to.deep.equal([[2], [[[4, 6]]], [8]]);
        });

        it('should map over objects preserving keys', () => {
            const input = {
                level1: {
                    level2: [1, 2, 3],
                    value: 42
                }
            };
            const result = treeAlgorithms.mapStructure(x => x * 2, input);
            
            expect(result.level1.level2).to.deep.equal([2, 4, 6]);
            expect(result.level1.value).to.equal(84);
        });

        it('should provide path information to mapping function', () => {
            const input = { a: { b: [1, 2] } };
            const paths = [];
            
            treeAlgorithms.mapStructure((value, path) => {
                paths.push([...path]);
                return value;
            }, input);
            
            expect(paths).to.deep.include(['a', 'b', 0]);
            expect(paths).to.deep.include(['a', 'b', 1]);
        });

        it('should handle Maps and Sets correctly', () => {
            const input = new Map([
                ['key1', [1, 2]],
                ['key2', new Set([3, 4])]
            ]);
            
            const result = treeAlgorithms.mapStructure(x => x * 10, input);
            
            expect(result).to.be.instanceOf(Map);
            expect(result.get('key1')).to.deep.equal([10, 20]);
            expect(Array.from(result.get('key2'))).to.include.members([30, 40]);
        });
    });

    describe('Filter Structure Operation', () => {
        it('should filter values while preserving structure', () => {
            const input = [[1, 2], [3, [4, 5]], [6]];
            const result = treeAlgorithms.filterStructure(x => x % 2 === 0, input);
            
            const flattened = treeAlgorithms.flatten(result);
            expect(flattened).to.deep.equal([2, 4, 6]);
        });

        it('should remove empty containers after filtering', () => {
            const input = {
                keep: [2, 4],
                remove: [1, 3, 5],
                mixed: [1, 2, 3, 4]
            };
            const result = treeAlgorithms.filterStructure(x => x % 2 === 0, input);
            
            expect(result).to.have.property('keep');
            expect(result).to.not.have.property('remove');
            expect(result).to.have.property('mixed');
            expect(result.mixed).to.deep.equal([2, 4]);
        });

        it('should provide path information to filter predicate', () => {
            const input = { section1: [1, 2], section2: [3, 4] };
            const result = treeAlgorithms.filterStructure((value, path) => {
                return path[0] === 'section1';
            }, input);
            
            expect(result).to.have.property('section1');
            expect(result).to.not.have.property('section2');
        });

        it('should return appropriate empty structure for no matches', () => {
            const arrayInput = [1, 3, 5];
            const objectInput = { a: 1, b: 3 };
            
            const arrayResult = treeAlgorithms.filterStructure(x => x % 2 === 0, arrayInput);
            const objectResult = treeAlgorithms.filterStructure(x => x % 2 === 0, objectInput);
            
            expect(arrayResult).to.be.an('array').that.is.empty;
            expect(objectResult).to.be.an('object').that.is.empty;
        });
    });

    describe('Reduce Structure Operation', () => {
        it('should reduce all values to single result', () => {
            const input = [[1, 2], [3, [4, 5]]];
            const result = treeAlgorithms.reduceStructure((acc, val) => acc + val, 0, input);
            
            expect(result).to.equal(15); // 1 + 2 + 3 + 4 + 5
        });

        it('should provide path information to reducer function', () => {
            const input = { a: [1, 2], b: { c: 3 } };
            const paths = [];
            
            treeAlgorithms.reduceStructure((acc, val, path) => {
                paths.push([...path]);
                return acc + val;
            }, 0, input);
            
            expect(paths).to.deep.include(['a', 0]);
            expect(paths).to.deep.include(['a', 1]);
            expect(paths).to.deep.include(['b', 'c']);
        });

        it('should work with complex accumulator types', () => {
            const input = { users: ['alice', 'bob'], admins: ['charlie'] };
            const result = treeAlgorithms.reduceStructure(
                (acc, val, path) => {
                    const role = path[0];
                    if (!acc[role]) acc[role] = [];
                    acc[role].push(val);
                    return acc;
                },
                {},
                input
            );
            
            expect(result.users).to.deep.equal(['alice', 'bob']);
            expect(result.admins).to.deep.equal(['charlie']);
        });
    });

    describe('Path Operations', () => {
        let testStructure;

        beforeEach(() => {
            testStructure = {
                level1: {
                    level2: {
                        array: [10, 20, 30],
                        value: 'test'
                    }
                },
                topLevel: 42
            };
        });

        describe('Get Path', () => {
            it('should retrieve values at specified paths', () => {
                expect(treeAlgorithms.getPath(testStructure, ['level1', 'level2', 'value']))
                    .to.equal('test');
                expect(treeAlgorithms.getPath(testStructure, ['level1', 'level2', 'array', 1]))
                    .to.equal(20);
                expect(treeAlgorithms.getPath(testStructure, ['topLevel']))
                    .to.equal(42);
            });

            it('should return undefined for invalid paths', () => {
                expect(treeAlgorithms.getPath(testStructure, ['nonexistent'])).to.be.undefined;
                expect(treeAlgorithms.getPath(testStructure, ['level1', 'level2', 'array', 10]))
                    .to.be.undefined;
                expect(treeAlgorithms.getPath(testStructure, ['level1', 'nonexistent']))
                    .to.be.undefined;
            });

            it('should handle edge cases gracefully', () => {
                expect(treeAlgorithms.getPath(null, ['any'])).to.be.undefined;
                expect(treeAlgorithms.getPath(testStructure, [])).to.equal(testStructure);
            });
        });

        describe('Set Path', () => {
            it('should set values at specified paths immutably', () => {
                const result = treeAlgorithms.setPath(testStructure, ['level1', 'level2', 'value'], 'new value');
                
                // Original should be unchanged
                expect(testStructure.level1.level2.value).to.equal('test');
                
                // Result should have new value
                expect(result.level1.level2.value).to.equal('new value');
                
                // Other values should be preserved
                expect(result.topLevel).to.equal(42);
            });

            it('should handle array indices correctly', () => {
                const result = treeAlgorithms.setPath(testStructure, ['level1', 'level2', 'array', 1], 99);
                
                expect(testStructure.level1.level2.array[1]).to.equal(20);
                expect(result.level1.level2.array[1]).to.equal(99);
                expect(result.level1.level2.array[0]).to.equal(10);
            });

            it('should return original for invalid paths', () => {
                const result = treeAlgorithms.setPath(testStructure, ['nonexistent', 'path'], 'value');
                
                expect(result).to.deep.equal(testStructure);
            });

            it('should handle empty path by returning the new value', () => {
                const result = treeAlgorithms.setPath(testStructure, [], 'completely new');
                
                expect(result).to.equal('completely new');
            });
        });

        describe('Get All Paths', () => {
            it('should return all paths to leaf values', () => {
                const paths = treeAlgorithms.getAllPaths(testStructure);
                
                const pathStrings = paths.map(p => p.keys.join('.'));
                expect(pathStrings).to.include('level1.level2.array.0');
                expect(pathStrings).to.include('level1.level2.array.1');
                expect(pathStrings).to.include('level1.level2.array.2');
                expect(pathStrings).to.include('level1.level2.value');
                expect(pathStrings).to.include('topLevel');
            });

            it('should include correct values for each path', () => {
                const paths = treeAlgorithms.getAllPaths(testStructure);
                
                const valuePath = paths.find(p => p.keys.join('.') === 'level1.level2.value');
                expect(valuePath.value).to.equal('test');
                
                const arrayPath = paths.find(p => p.keys.join('.') === 'level1.level2.array.1');
                expect(arrayPath.value).to.equal(20);
            });
        });
    });

    describe('Structure Analysis', () => {
        describe('Get Depth', () => {
            it('should calculate correct depth for nested structures', () => {
                expect(treeAlgorithms.getDepth(42)).to.equal(0);
                expect(treeAlgorithms.getDepth([1, 2, 3])).to.equal(1);
                expect(treeAlgorithms.getDepth([[1, 2], [3]])).to.equal(2);
                expect(treeAlgorithms.getDepth({ a: { b: { c: 1 } } })).to.equal(3);
            });

            it('should handle mixed structure types', () => {
                const mixed = {
                    array: [1, [2, [3]]],
                    object: { nested: { deep: 'value' } }
                };
                
                expect(treeAlgorithms.getDepth(mixed)).to.equal(4); // a.array.[1].[1]
            });

            it('should handle Maps and Sets', () => {
                const mapStructure = new Map([
                    ['key', new Set([1, new Map([['inner', 'value']])])]
                ]);
                
                expect(treeAlgorithms.getDepth(mapStructure)).to.equal(3);
            });
        });

        describe('Count Operations', () => {
            let testStructure;

            beforeEach(() => {
                testStructure = {
                    numbers: [1, 2, 3],
                    nested: {
                        more: [4, 5]
                    },
                    single: 6
                };
            });

            it('should count leaf nodes correctly', () => {
                const leafCount = treeAlgorithms.getLeafCount(testStructure);
                expect(leafCount).to.equal(6); // 1, 2, 3, 4, 5, 6
            });

            it('should count all nodes including containers', () => {
                const totalCount = treeAlgorithms.getNodeCount(testStructure);
                // Root + numbers array + 3 items + nested object + more array + 2 items + single
                expect(totalCount).to.be.greaterThan(6);
            });
        });
    });

    describe('Find Substructures', () => {
        let complexStructure;

        beforeEach(() => {
            complexStructure = {
                components: [
                    { type: 'button', id: 'btn1' },
                    { type: 'input', id: 'inp1' }
                ],
                layouts: {
                    main: { type: 'container', children: 3 },
                    sidebar: { type: 'container', children: 1 }
                },
                config: { theme: 'dark', type: 'config' }
            };
        });

        it('should find structures matching predicate', () => {
            const containers = treeAlgorithms.findSubstructures(
                complexStructure,
                (item) => typeof item === 'object' && item.type === 'container'
            );
            
            expect(containers).to.have.lengthOf(2);
            expect(containers[0].value.children).to.exist;
            expect(containers[1].value.children).to.exist;
        });

        it('should provide path information for found structures', () => {
            const buttons = treeAlgorithms.findSubstructures(
                complexStructure,
                (item, path) => typeof item === 'object' && item.type === 'button'
            );
            
            expect(buttons).to.have.lengthOf(1);
            expect(buttons[0].keys).to.deep.equal(['components', 0]);
        });

        it('should find leaf values when specified', () => {
            const strings = treeAlgorithms.findSubstructures(
                complexStructure,
                (item) => typeof item === 'string'
            );
            
            const stringValues = strings.map(s => s.value);
            expect(stringValues).to.include('button');
            expect(stringValues).to.include('input');
            expect(stringValues).to.include('dark');
        });
    });

    describe('Diff Operation', () => {
        let oldStructure, newStructure;

        beforeEach(() => {
            oldStructure = {
                a: 1,
                b: [2, 3],
                c: { d: 4 }
            };
            
            newStructure = {
                a: 10, // modified
                b: [2, 3, 5], // added item
                c: { d: 4 }, // unchanged
                e: 6 // added property
            };
        });

        it('should detect added elements', () => {
            const diff = treeAlgorithms.diff(oldStructure, newStructure);
            
            expect(diff.added).to.have.lengthOf(2);
            
            const addedPaths = diff.added.map(a => a.keys.join('.'));
            expect(addedPaths).to.include('b.2');
            expect(addedPaths).to.include('e');
        });

        it('should detect modified elements', () => {
            const diff = treeAlgorithms.diff(oldStructure, newStructure);
            
            expect(diff.modified).to.have.lengthOf(1);
            expect(diff.modified[0].path.keys).to.deep.equal(['a']);
            expect(diff.modified[0].oldValue).to.equal(1);
            expect(diff.modified[0].newValue).to.equal(10);
        });

        it('should detect removed elements', () => {
            const structureWithRemoval = { a: 1, c: { d: 4 } }; // removed 'b'
            const diff = treeAlgorithms.diff(oldStructure, structureWithRemoval);
            
            expect(diff.removed).to.have.lengthOf(1);
            expect(diff.removed[0].keys).to.deep.equal(['b']);
        });

        it('should handle complex nested changes', () => {
            const complexOld = { users: [{ name: 'alice' }, { name: 'bob' }] };
            const complexNew = { users: [{ name: 'alice', active: true }, { name: 'charlie' }] };
            
            const diff = treeAlgorithms.diff(complexOld, complexNew);
            
            // Should detect user modifications
            expect(diff.added.length + diff.modified.length).to.be.greaterThan(0);
        });
    });

    describe('Merge Operation', () => {
        let baseStructure, overrideStructure;

        beforeEach(() => {
            baseStructure = {
                config: { theme: 'light', lang: 'en' },
                features: ['feature1', 'feature2'],
                version: '1.0.0'
            };
            
            overrideStructure = {
                config: { theme: 'dark' }, // partial override
                features: ['feature3'], // array replacement
                newConfig: 'added'
            };
        });

        it('should merge with override strategy', () => {
            const result = treeAlgorithms.merge(baseStructure, overrideStructure, 'override');
            
            expect(result.config.theme).to.equal('dark');
            expect(result.config.lang).to.equal('en'); // preserved from base
            expect(result.features).to.deep.equal(['feature3']); // completely replaced
            expect(result.version).to.equal('1.0.0'); // preserved from base
            expect(result.newConfig).to.equal('added'); // added from override
        });

        it('should merge arrays with concat strategy', () => {
            const result = treeAlgorithms.merge(baseStructure, overrideStructure, 'concat');
            
            expect(result.features).to.deep.equal(['feature1', 'feature2', 'feature3']);
        });

        it('should merge recursively with merge strategy', () => {
            const result = treeAlgorithms.merge(baseStructure, overrideStructure, 'merge');
            
            expect(result.config.theme).to.equal('dark');
            expect(result.config.lang).to.equal('en');
            expect(result.features).to.deep.equal(['feature1', 'feature3']); // array merge at indices
        });

        it('should preserve original structures (immutability)', () => {
            const result = treeAlgorithms.merge(baseStructure, overrideStructure);
            
            expect(baseStructure.config.theme).to.equal('light');
            expect(overrideStructure.config).to.not.have.property('lang');
            expect(result.config.theme).to.equal('dark');
            expect(result.config.lang).to.equal('en');
        });
    });

    describe('Performance and Scalability', () => {
        it('should handle deep nesting efficiently', () => {
            // Create deeply nested structure
            let deep = 'value';
            for (let i = 0; i < 100; i++) {
                deep = { [`level${i}`]: deep };
            }
            
            const startTime = Date.now();
            const depth = treeAlgorithms.getDepth(deep);
            const endTime = Date.now();
            
            expect(depth).to.equal(100);
            expect(endTime - startTime).to.be.lessThan(100); // Should be fast
        });

        it('should handle wide structures efficiently', () => {
            // Create structure with many properties
            const wide = {};
            for (let i = 0; i < 1000; i++) {
                wide[`prop${i}`] = i;
            }
            
            const startTime = Date.now();
            const flattened = treeAlgorithms.flatten(wide);
            const endTime = Date.now();
            
            expect(flattened).to.have.lengthOf(1000);
            expect(endTime - startTime).to.be.lessThan(100);
        });

        it('should handle large array operations efficiently', () => {
            const largeArray = Array.from({ length: 10000 }, (_, i) => [i, i * 2]);
            
            const startTime = Date.now();
            const result = treeAlgorithms.mapStructure(x => x + 1, largeArray);
            const endTime = Date.now();
            
            expect(result).to.have.lengthOf(10000);
            expect(endTime - startTime).to.be.lessThan(1000);
        });
    });

    describe('Integration with SAP Standards', () => {
        it('should handle SAP entity structures', () => {
            const sapEntity = {
                BusinessPartner: {
                    ID: 'BP001',
                    Name: 'Test Partner',
                    Addresses: [
                        { Type: 'Billing', Street: 'Main St' },
                        { Type: 'Shipping', Street: 'Oak Ave' }
                    ]
                }
            };
            
            const addresses = treeAlgorithms.findSubstructures(
                sapEntity,
                (item, path) => path.includes('Addresses') && typeof item === 'object' && item.Type
            );
            
            expect(addresses).to.have.lengthOf(2);
            expect(addresses[0].value.Type).to.be.oneOf(['Billing', 'Shipping']);
        });

        it('should support CAP-style navigation properties', () => {
            const capStructure = {
                Orders: [
                    {
                        ID: 'ORD001',
                        Items: [
                            { Product: 'PRD001', Quantity: 5 },
                            { Product: 'PRD002', Quantity: 3 }
                        ]
                    }
                ]
            };
            
            const products = treeAlgorithms.findSubstructures(
                capStructure,
                (item, path) => path.includes('Items') && typeof item === 'object' && item.Product
            );
            
            expect(products).to.have.lengthOf(2);
            expect(products.map(p => p.value.Product)).to.include.members(['PRD001', 'PRD002']);
        });

        it('should handle i18n resource bundle structures', () => {
            const i18nBundle = {
                en: {
                    buttons: { save: 'Save', cancel: 'Cancel' },
                    messages: { error: 'An error occurred' }
                },
                de: {
                    buttons: { save: 'Speichern', cancel: 'Abbrechen' },
                    messages: { error: 'Ein Fehler ist aufgetreten' }
                }
            };
            
            const germanTexts = treeAlgorithms.filterStructure(
                (item, path) => path[0] === 'de' && typeof item === 'string',
                i18nBundle
            );
            
            const flattened = treeAlgorithms.flatten(germanTexts);
            expect(flattened).to.include('Speichern');
            expect(flattened).to.include('Ein Fehler ist aufgetreten');
        });
    });

    describe('Error Handling and Edge Cases', () => {
        it('should handle null and undefined gracefully', () => {
            expect(treeAlgorithms.flatten(null)).to.deep.equal([]);
            expect(treeAlgorithms.flatten(undefined)).to.deep.equal([]);
            
            expect(treeAlgorithms.getDepth(null)).to.equal(0);
            expect(treeAlgorithms.getDepth(undefined)).to.equal(0);
        });

        it('should handle circular references safely', () => {
            const circular = { a: 1 };
            circular.self = circular;
            
            // These operations should not cause infinite loops
            expect(() => {
                treeAlgorithms.getDepth(circular);
            }).to.not.throw();
            
            expect(() => {
                treeAlgorithms.getLeafCount(circular);
            }).to.not.throw();
        });

        it('should maintain type safety for different container types', () => {
            const mapResult = treeAlgorithms.mapStructure(x => x * 2, new Map([['a', 1]]));
            expect(mapResult).to.be.instanceOf(Map);
            
            const setResult = treeAlgorithms.mapStructure(x => x * 2, new Set([1, 2]));
            expect(setResult).to.be.instanceOf(Set);
            
            const arrayResult = treeAlgorithms.mapStructure(x => x * 2, [1, 2]);
            expect(arrayResult).to.be.an('array');
        });
    });
});