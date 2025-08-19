const cds = require('@sap/cds');
const BaseService = require('../utils/BaseService');

/**
 * SAP-compliant Tree Algorithms implementation
 * Provides hierarchical data structure manipulation for code analysis
 */
class TreeAlgorithms extends BaseService {
    constructor() {
        super();
        this.logger = cds.log('tree-algorithms');
    }

    /**
     * Initialize the tree algorithms service
     */
    async initializeService() {
        this.logger.info('Initializing Tree Algorithms Service');
    }

    /**
     * Check if a value is a leaf node
     */
    isLeaf(value) {
        if (value === null || value === undefined) return true;
        if (typeof value === 'object') {
            if (Array.isArray(value)) return value.length === 0;
            if (value instanceof Map) return value.size === 0;
            if (value instanceof Set) return value.size === 0;
            return Object.keys(value).length === 0;
        }
        return true;
    }

    /**
     * Flatten a nested structure into a flat array
     */
    flatten(structure) {
        const result = [];

        const flattenRecursive = (item) => {
            if (this.isLeaf(item)) {
                if (item !== null && item !== undefined) {
                    result.push(item);
                }
            } else if (Array.isArray(item)) {
                item.forEach(flattenRecursive);
            } else if (item instanceof Map) {
                item.forEach((value) => flattenRecursive(value));
            } else if (item instanceof Set) {
                item.forEach((value) => flattenRecursive(value));
            } else if (typeof item === 'object') {
                Object.values(item).forEach(flattenRecursive);
            }
        };

        flattenRecursive(structure);
        return result;
    }

    /**
     * Map a function over a nested structure while preserving its shape
     */
    mapStructure(fn, structure) {
        const mapRecursive = (item, path = []) => {
            if (this.isLeaf(item)) {
                return fn(item, path);
            } else if (Array.isArray(item)) {
                return item.map((val, index) => 
                    mapRecursive(val, [...path, index])
                );
            } else if (item instanceof Map) {
                const result = new Map();
                item.forEach((value, key) => {
                    result.set(key, mapRecursive(value, [...path, key]));
                });
                return result;
            } else if (item instanceof Set) {
                const result = new Set();
                let index = 0;
                item.forEach((value) => {
                    result.add(mapRecursive(value, [...path, `set-item-${index++}`]));
                });
                return result;
            } else if (typeof item === 'object') {
                const result = {};
                Object.entries(item).forEach(([key, value]) => {
                    result[key] = mapRecursive(value, [...path, key]);
                });
                return result;
            }
            return item;
        };

        return mapRecursive(structure);
    }

    /**
     * Filter a nested structure based on a predicate
     */
    filterStructure(predicate, structure) {
        const filterRecursive = (item, path = []) => {
            if (this.isLeaf(item)) {
                return predicate(item, path) ? item : undefined;
            } else if (Array.isArray(item)) {
                const filtered = item
                    .map((val, index) => filterRecursive(val, [...path, index]))
                    .filter(val => val !== undefined);
                return filtered.length > 0 ? filtered : undefined;
            } else if (item instanceof Map) {
                const result = new Map();
                item.forEach((value, key) => {
                    const filtered = filterRecursive(value, [...path, key]);
                    if (filtered !== undefined) {
                        result.set(key, filtered);
                    }
                });
                return result.size > 0 ? result : undefined;
            } else if (item instanceof Set) {
                const result = new Set();
                let index = 0;
                item.forEach((value) => {
                    const filtered = filterRecursive(value, [...path, `set-item-${index++}`]);
                    if (filtered !== undefined) {
                        result.add(filtered);
                    }
                });
                return result.size > 0 ? result : undefined;
            } else if (typeof item === 'object') {
                const result = {};
                let hasValues = false;
                Object.entries(item).forEach(([key, value]) => {
                    const filtered = filterRecursive(value, [...path, key]);
                    if (filtered !== undefined) {
                        result[key] = filtered;
                        hasValues = true;
                    }
                });
                return hasValues ? result : undefined;
            }
            return undefined;
        };

        const result = filterRecursive(structure);
        return result !== undefined ? result : (Array.isArray(structure) ? [] : {});
    }

    /**
     * Reduce a nested structure to a single value
     */
    reduceStructure(fn, initial, structure) {
        let accumulator = initial;

        const reduceRecursive = (item, path = []) => {
            if (this.isLeaf(item)) {
                accumulator = fn(accumulator, item, path);
            } else if (Array.isArray(item)) {
                item.forEach((val, index) => 
                    reduceRecursive(val, [...path, index])
                );
            } else if (item instanceof Map) {
                item.forEach((value, key) => {
                    reduceRecursive(value, [...path, key]);
                });
            } else if (item instanceof Set) {
                let index = 0;
                item.forEach((value) => {
                    reduceRecursive(value, [...path, `set-item-${index++}`]);
                });
            } else if (typeof item === 'object') {
                Object.entries(item).forEach(([key, value]) => {
                    reduceRecursive(value, [...path, key]);
                });
            }
        };

        reduceRecursive(structure);
        return accumulator;
    }

    /**
     * Get value at a specific path in the structure
     */
    getPath(structure, path) {
        let current = structure;

        for (const key of path) {
            if (current === null || current === undefined) {
                return undefined;
            }

            if (Array.isArray(current)) {
                const index = typeof key === 'number' ? key : parseInt(key, 10);
                if (isNaN(index) || index < 0 || index >= current.length) {
                    return undefined;
                }
                current = current[index];
            } else if (current instanceof Map) {
                if (!current.has(key)) {
                    return undefined;
                }
                current = current.get(key);
            } else if (current instanceof Set) {
                // Sets don't have indexed access, convert to array
                const arr = Array.from(current);
                const index = typeof key === 'number' ? key : parseInt(key, 10);
                if (isNaN(index) || index < 0 || index >= arr.length) {
                    return undefined;
                }
                current = arr[index];
            } else if (typeof current === 'object') {
                if (!(key in current)) {
                    return undefined;
                }
                current = current[key];
            } else {
                return undefined;
            }
        }

        return current;
    }

    /**
     * Set value at a specific path in the structure (immutable)
     */
    setPath(structure, path, value) {
        if (path.length === 0) {
            return value;
        }

        const deepClone = (obj) => {
            if (obj === null || typeof obj !== 'object') return obj;
            if (obj instanceof Date) return new Date(obj);
            if (obj instanceof Array) return obj.map(item => deepClone(item));
            if (obj instanceof Map) {
                const map = new Map();
                obj.forEach((val, key) => map.set(key, deepClone(val)));
                return map;
            }
            if (obj instanceof Set) {
                const set = new Set();
                obj.forEach(val => set.add(deepClone(val)));
                return set;
            }
            const cloned = {};
            Object.keys(obj).forEach(key => {
                cloned[key] = deepClone(obj[key]);
            });
            return cloned;
        };

        const result = deepClone(structure);
        let current = result;
        
        for (let i = 0; i < path.length - 1; i++) {
            const key = path[i];
            
            if (Array.isArray(current)) {
                const index = typeof key === 'number' ? key : parseInt(key, 10);
                if (!isNaN(index) && index >= 0 && index < current.length) {
                    current = current[index];
                } else {
                    return result; // Invalid path
                }
            } else if (current instanceof Map) {
                current = current.get(key);
            } else if (typeof current === 'object') {
                current = current[key];
            }
        }

        const lastKey = path[path.length - 1];
        if (Array.isArray(current)) {
            const index = typeof lastKey === 'number' ? lastKey : parseInt(lastKey, 10);
            if (!isNaN(index) && index >= 0 && index < current.length) {
                current[index] = value;
            }
        } else if (current instanceof Map) {
            current.set(lastKey, value);
        } else if (typeof current === 'object') {
            current[lastKey] = value;
        }

        return result;
    }

    /**
     * Get all paths in the structure
     */
    getAllPaths(structure) {
        const paths = [];

        const collectPaths = (item, currentPath = []) => {
            if (this.isLeaf(item)) {
                paths.push({
                    keys: currentPath,
                    value: item
                });
            } else if (Array.isArray(item)) {
                item.forEach((val, index) => {
                    collectPaths(val, [...currentPath, index]);
                });
            } else if (item instanceof Map) {
                item.forEach((value, key) => {
                    collectPaths(value, [...currentPath, key]);
                });
            } else if (item instanceof Set) {
                let index = 0;
                item.forEach((value) => {
                    collectPaths(value, [...currentPath, `set-item-${index++}`]);
                });
            } else if (typeof item === 'object') {
                Object.entries(item).forEach(([key, value]) => {
                    collectPaths(value, [...currentPath, key]);
                });
            }
        };

        collectPaths(structure);
        return paths;
    }

    /**
     * Get the depth of the structure
     */
    getDepth(structure) {
        if (this.isLeaf(structure)) {
            return 0;
        }

        let maxDepth = 0;

        if (Array.isArray(structure)) {
            structure.forEach(item => {
                maxDepth = Math.max(maxDepth, this.getDepth(item));
            });
        } else if (structure instanceof Map) {
            structure.forEach(value => {
                maxDepth = Math.max(maxDepth, this.getDepth(value));
            });
        } else if (structure instanceof Set) {
            structure.forEach(value => {
                maxDepth = Math.max(maxDepth, this.getDepth(value));
            });
        } else if (typeof structure === 'object') {
            Object.values(structure).forEach(value => {
                maxDepth = Math.max(maxDepth, this.getDepth(value));
            });
        }

        return maxDepth + 1;
    }

    /**
     * Count all leaf nodes in the structure
     */
    getLeafCount(structure) {
        return this.reduceStructure(
            (count, item) => count + 1,
            0,
            structure
        );
    }

    /**
     * Count all nodes (including internal nodes) in the structure
     */
    getNodeCount(structure) {
        let count = 1; // Count the current node

        if (!this.isLeaf(structure)) {
            if (Array.isArray(structure)) {
                structure.forEach(item => {
                    count += this.getNodeCount(item);
                });
            } else if (structure instanceof Map) {
                structure.forEach(value => {
                    count += this.getNodeCount(value);
                });
            } else if (structure instanceof Set) {
                structure.forEach(value => {
                    count += this.getNodeCount(value);
                });
            } else if (typeof structure === 'object') {
                Object.values(structure).forEach(value => {
                    count += this.getNodeCount(value);
                });
            }
        }

        return count;
    }

    /**
     * Find all substructures matching a predicate
     */
    findSubstructures(structure, predicate) {
        const matches = [];

        const searchRecursive = (item, path = []) => {
            if (predicate(item, path)) {
                matches.push({
                    keys: path,
                    value: item
                });
            }

            if (!this.isLeaf(item)) {
                if (Array.isArray(item)) {
                    item.forEach((val, index) => {
                        searchRecursive(val, [...path, index]);
                    });
                } else if (item instanceof Map) {
                    item.forEach((value, key) => {
                        searchRecursive(value, [...path, key]);
                    });
                } else if (item instanceof Set) {
                    let index = 0;
                    item.forEach((value) => {
                        searchRecursive(value, [...path, `set-item-${index++}`]);
                    });
                } else if (typeof item === 'object') {
                    Object.entries(item).forEach(([key, value]) => {
                        searchRecursive(value, [...path, key]);
                    });
                }
            }
        };

        searchRecursive(structure);
        return matches;
    }

    /**
     * Diff two structures and return the differences
     */
    diff(oldStructure, newStructure, path = []) {
        const differences = {
            added: [],
            removed: [],
            modified: []
        };

        const compareStructures = (oldVal, newVal, currentPath) => {
            // Both are leaves
            if (this.isLeaf(oldVal) && this.isLeaf(newVal)) {
                if (oldVal !== newVal) {
                    differences.modified.push({
                        path: { keys: currentPath },
                        oldValue: oldVal,
                        newValue: newVal
                    });
                }
                return;
            }

            // One is leaf, other is not
            if (this.isLeaf(oldVal) || this.isLeaf(newVal)) {
                differences.modified.push({
                    path: { keys: currentPath },
                    oldValue: oldVal,
                    newValue: newVal
                });
                return;
            }

            // Both are objects/arrays
            if (Array.isArray(oldVal) && Array.isArray(newVal)) {
                const maxLength = Math.max(oldVal.length, newVal.length);
                for (let i = 0; i < maxLength; i++) {
                    if (i >= oldVal.length) {
                        differences.added.push({
                            keys: [...currentPath, i],
                            value: newVal[i]
                        });
                    } else if (i >= newVal.length) {
                        differences.removed.push({
                            keys: [...currentPath, i],
                            value: oldVal[i]
                        });
                    } else {
                        compareStructures(oldVal[i], newVal[i], [...currentPath, i]);
                    }
                }
            } else if (typeof oldVal === 'object' && typeof newVal === 'object') {
                const allKeys = new Set([
                    ...Object.keys(oldVal),
                    ...Object.keys(newVal)
                ]);

                allKeys.forEach(key => {
                    if (!(key in oldVal)) {
                        differences.added.push({
                            keys: [...currentPath, key],
                            value: newVal[key]
                        });
                    } else if (!(key in newVal)) {
                        differences.removed.push({
                            keys: [...currentPath, key],
                            value: oldVal[key]
                        });
                    } else {
                        compareStructures(oldVal[key], newVal[key], [...currentPath, key]);
                    }
                });
            } else {
                // Different types
                differences.modified.push({
                    path: { keys: currentPath },
                    oldValue: oldVal,
                    newValue: newVal
                });
            }
        };

        compareStructures(oldStructure, newStructure, path);
        return differences;
    }

    /**
     * Merge two structures with custom merge strategy
     */
    merge(base, override, mergeStrategy = 'override') {
        if (this.isLeaf(base) || this.isLeaf(override)) {
            return mergeStrategy === 'override' ? override : base;
        }

        if (Array.isArray(base) && Array.isArray(override)) {
            switch (mergeStrategy) {
                case 'override':
                    return override;
                case 'concat':
                    return [...base, ...override];
                case 'merge':
                    const merged = [...base];
                    override.forEach((item, index) => {
                        if (index < merged.length) {
                            merged[index] = this.merge(merged[index], item, mergeStrategy);
                        } else {
                            merged.push(item);
                        }
                    });
                    return merged;
                default:
                    return override;
            }
        }

        if (typeof base === 'object' && typeof override === 'object') {
            const result = { ...base };
            Object.entries(override).forEach(([key, value]) => {
                if (key in result) {
                    result[key] = this.merge(result[key], value, mergeStrategy);
                } else {
                    result[key] = value;
                }
            });
            return result;
        }

        return override;
    }
}

module.exports = TreeAlgorithms;