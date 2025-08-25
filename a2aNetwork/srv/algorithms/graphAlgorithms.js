const cds = require('@sap/cds');
const BaseService = require('../utils/BaseService');

/**
 * SAP-compliant Graph Algorithms implementation following CLRS specifications
 * Provides advanced graph analysis capabilities for code dependency analysis
 */
class GraphAlgorithms extends BaseService {
    constructor() {
        super();
        this.logger = cds.log('graph-algorithms');
    }

    /**
     * Initialize the graph algorithms service
     */
    async initializeService() {
        this.logger.info('Initializing Graph Algorithms Service');
    }

    /**
     * Graph representation following SAP naming conventions
     */
    createGraph() {
        return {
            nodes: new Set(),
            edges: new Map(),
            weights: new Map(),
            metadata: new Map()
        };
    }

    /**
     * Add node to graph with SAP-compliant error handling
     */
    addNode(graph, nodeId, metadata = {}) {
        if (!nodeId) {
            throw new Error('Node ID is required');
        }

        graph.nodes.add(nodeId);
        graph.metadata.set(nodeId, {
            ...metadata,
            addedAt: new Date().toISOString()
        });

        if (!graph.edges.has(nodeId)) {
            graph.edges.set(nodeId, new Set());
        }

        return graph;
    }

    /**
     * Add edge to graph with optional weight
     */
    addEdge(graph, fromNodeId, toNodeId, weight = 1) {
        this.addNode(graph, fromNodeId);
        this.addNode(graph, toNodeId);

        graph.edges.get(fromNodeId).add(toNodeId);

        if (!graph.weights.has(fromNodeId)) {
            graph.weights.set(fromNodeId, new Map());
        }
        graph.weights.get(fromNodeId).set(toNodeId, weight);

        return graph;
    }

    /**
     * Depth-First Search (DFS) implementation
     * @returns {Object} Traversal result with visit order and metadata
     */
    depthFirstSearch(graph, startNodeId) {
        const visited = new Set();
        const visitOrder = [];
        const discovered = new Map();
        const finished = new Map();
        const predecessors = new Map();
        let time = 0;

        const dfsVisit = (nodeId) => {
            visited.add(nodeId);
            visitOrder.push(nodeId);
            discovered.set(nodeId, ++time);

            const neighbors = graph.edges.get(nodeId) || new Set();
            for (const neighbor of neighbors) {
                if (!visited.has(neighbor)) {
                    predecessors.set(neighbor, nodeId);
                    dfsVisit(neighbor);
                }
            }

            finished.set(nodeId, ++time);
        };

        // Start DFS from the given node
        if (graph.nodes.has(startNodeId)) {
            dfsVisit(startNodeId);
        }

        // Visit any remaining unvisited nodes
        for (const nodeId of graph.nodes) {
            if (!visited.has(nodeId)) {
                dfsVisit(nodeId);
            }
        }

        return {
            visitOrder,
            discovered,
            finished,
            predecessors,
            hasPath: (targetNodeId) => visited.has(targetNodeId)
        };
    }

    /**
     * Breadth-First Search (BFS) implementation
     */
    breadthFirstSearch(graph, startNodeId) {
        const visited = new Set();
        const visitOrder = [];
        const distances = new Map();
        const predecessors = new Map();
        const queue = [];

        if (!graph.nodes.has(startNodeId)) {
            return {
                visitOrder: [],
                distances: new Map(),
                predecessors: new Map(),
                hasPath: () => false
            };
        }

        // Initialize start node
        visited.add(startNodeId);
        visitOrder.push(startNodeId);
        distances.set(startNodeId, 0);
        predecessors.set(startNodeId, null);
        queue.push(startNodeId);

        while (queue.length > 0) {
            const currentNodeId = queue.shift();
            const neighbors = graph.edges.get(currentNodeId) || new Set();

            for (const neighbor of neighbors) {
                if (!visited.has(neighbor)) {
                    visited.add(neighbor);
                    visitOrder.push(neighbor);
                    distances.set(neighbor, distances.get(currentNodeId) + 1);
                    predecessors.set(neighbor, currentNodeId);
                    queue.push(neighbor);
                }
            }
        }

        return {
            visitOrder,
            distances,
            predecessors,
            hasPath: (targetNodeId) => visited.has(targetNodeId),
            getPath: (targetNodeId) => this._reconstructPath(predecessors, startNodeId, targetNodeId)
        };
    }

    /**
     * Topological Sort using DFS
     * For directed acyclic graphs (DAGs)
     */
    topologicalSort(graph) {
        const visited = new Set();
        const stack = [];
        const hasCycle = { value: false };

        const dfsVisit = (nodeId, recursionStack = new Set()) => {
            visited.add(nodeId);
            recursionStack.add(nodeId);

            const neighbors = graph.edges.get(nodeId) || new Set();
            for (const neighbor of neighbors) {
                if (!visited.has(neighbor)) {
                    dfsVisit(neighbor, recursionStack);
                } else if (recursionStack.has(neighbor)) {
                    hasCycle.value = true;
                }
            }

            recursionStack.delete(nodeId);
            stack.push(nodeId);
        };

        // Visit all nodes
        for (const nodeId of graph.nodes) {
            if (!visited.has(nodeId)) {
                dfsVisit(nodeId);
            }
        }

        if (hasCycle.value) {
            throw new Error('Graph contains a cycle - topological sort not possible');
        }

        return stack.reverse();
    }

    /**
     * Dijkstra's shortest path algorithm
     */
    dijkstra(graph, sourceNodeId) {
        const distances = new Map();
        const predecessors = new Map();
        const visited = new Set();
        const priorityQueue = [];

        // Initialize distances
        for (const nodeId of graph.nodes) {
            distances.set(nodeId, nodeId === sourceNodeId ? 0 : Infinity);
            predecessors.set(nodeId, null);
        }

        priorityQueue.push({ nodeId: sourceNodeId, distance: 0 });

        while (priorityQueue.length > 0) {
            // Get node with minimum distance
            priorityQueue.sort((a, b) => a.distance - b.distance);
            const { nodeId: currentNodeId } = priorityQueue.shift();

            if (visited.has(currentNodeId)) continue;
            visited.add(currentNodeId);

            const neighbors = graph.edges.get(currentNodeId) || new Set();
            const weightMap = graph.weights.get(currentNodeId) || new Map();

            for (const neighbor of neighbors) {
                const weight = weightMap.get(neighbor) || 1;
                const altDistance = distances.get(currentNodeId) + weight;

                if (altDistance < distances.get(neighbor)) {
                    distances.set(neighbor, altDistance);
                    predecessors.set(neighbor, currentNodeId);
                    priorityQueue.push({ nodeId: neighbor, distance: altDistance });
                }
            }
        }

        return {
            distances,
            predecessors,
            hasPath: (targetNodeId) => distances.get(targetNodeId) < Infinity,
            getPath: (targetNodeId) => this._reconstructPath(predecessors, sourceNodeId, targetNodeId),
            getDistance: (targetNodeId) => distances.get(targetNodeId)
        };
    }

    /**
     * Bellman-Ford algorithm for shortest paths with negative weights
     */
    bellmanFord(graph, sourceNodeId) {
        const distances = new Map();
        const predecessors = new Map();

        // Initialize distances
        for (const nodeId of graph.nodes) {
            distances.set(nodeId, nodeId === sourceNodeId ? 0 : Infinity);
            predecessors.set(nodeId, null);
        }

        // Relax edges |V| - 1 times
        const nodeCount = graph.nodes.size;
        for (let i = 0; i < nodeCount - 1; i++) {
            for (const fromNodeId of graph.nodes) {
                const neighbors = graph.edges.get(fromNodeId) || new Set();
                const weightMap = graph.weights.get(fromNodeId) || new Map();

                for (const toNodeId of neighbors) {
                    const weight = weightMap.get(toNodeId) || 1;
                    if (distances.get(fromNodeId) + weight < distances.get(toNodeId)) {
                        distances.set(toNodeId, distances.get(fromNodeId) + weight);
                        predecessors.set(toNodeId, fromNodeId);
                    }
                }
            }
        }

        // Check for negative cycles
        let hasNegativeCycle = false;
        for (const fromNodeId of graph.nodes) {
            const neighbors = graph.edges.get(fromNodeId) || new Set();
            const weightMap = graph.weights.get(fromNodeId) || new Map();

            for (const toNodeId of neighbors) {
                const weight = weightMap.get(toNodeId) || 1;
                if (distances.get(fromNodeId) + weight < distances.get(toNodeId)) {
                    hasNegativeCycle = true;
                    break;
                }
            }
            if (hasNegativeCycle) break;
        }

        return {
            distances,
            predecessors,
            hasNegativeCycle,
            hasPath: (targetNodeId) => !hasNegativeCycle && distances.get(targetNodeId) < Infinity,
            getPath: (targetNodeId) => this._reconstructPath(predecessors, sourceNodeId, targetNodeId)
        };
    }

    /**
     * Find strongly connected components using Kosaraju's algorithm
     */
    findStronglyConnectedComponents(graph) {
        // First DFS to get finish times
        const visited = new Set();
        const finishOrder = [];

        const dfsVisit = (nodeId) => {
            visited.add(nodeId);
            const neighbors = graph.edges.get(nodeId) || new Set();
            for (const neighbor of neighbors) {
                if (!visited.has(neighbor)) {
                    dfsVisit(neighbor);
                }
            }
            finishOrder.push(nodeId);
        };

        // Visit all nodes
        for (const nodeId of graph.nodes) {
            if (!visited.has(nodeId)) {
                dfsVisit(nodeId);
            }
        }

        // Create transposed graph
        const transposedGraph = this._transposeGraph(graph);

        // Second DFS on transposed graph
        visited.clear();
        const components = [];

        const dfsVisitTransposed = (nodeId, component) => {
            visited.add(nodeId);
            component.push(nodeId);
            const neighbors = transposedGraph.edges.get(nodeId) || new Set();
            for (const neighbor of neighbors) {
                if (!visited.has(neighbor)) {
                    dfsVisitTransposed(neighbor, component);
                }
            }
        };

        // Process nodes in reverse finish order
        finishOrder.reverse();
        for (const nodeId of finishOrder) {
            if (!visited.has(nodeId)) {
                const component = [];
                dfsVisitTransposed(nodeId, component);
                components.push(component);
            }
        }

        return components;
    }

    /**
     * Detect cycles in the graph
     */
    detectCycles(graph) {
        const visited = new Set();
        const recursionStack = new Set();
        const cycles = [];

        const dfsVisit = (nodeId, path = []) => {
            visited.add(nodeId);
            recursionStack.add(nodeId);
            path.push(nodeId);

            const neighbors = graph.edges.get(nodeId) || new Set();
            for (const neighbor of neighbors) {
                if (!visited.has(neighbor)) {
                    dfsVisit(neighbor, [...path]);
                } else if (recursionStack.has(neighbor)) {
                    // Found a cycle
                    const cycleStartIndex = path.indexOf(neighbor);
                    if (cycleStartIndex !== -1) {
                        cycles.push([...path.slice(cycleStartIndex), neighbor]);
                    }
                }
            }

            recursionStack.delete(nodeId);
        };

        for (const nodeId of graph.nodes) {
            if (!visited.has(nodeId)) {
                dfsVisit(nodeId);
            }
        }

        return cycles;
    }

    /**
     * Private helper method to reconstruct path from predecessors
     */
    _reconstructPath(predecessors, sourceNodeId, targetNodeId) {
        const path = [];
        let current = targetNodeId;

        while (current !== null && current !== sourceNodeId) {
            path.unshift(current);
            current = predecessors.get(current);
        }

        if (current === sourceNodeId) {
            path.unshift(sourceNodeId);
            return path;
        }

        return []; // No path exists
    }

    /**
     * Private helper to transpose a graph
     */
    _transposeGraph(graph) {
        const transposed = this.createGraph();

        // Add all nodes
        for (const nodeId of graph.nodes) {
            transposed.nodes.add(nodeId);
            transposed.edges.set(nodeId, new Set());
        }

        // Reverse all edges
        for (const fromNodeId of graph.nodes) {
            const neighbors = graph.edges.get(fromNodeId) || new Set();
            for (const toNodeId of neighbors) {
                transposed.edges.get(toNodeId).add(fromNodeId);
            }
        }

        return transposed;
    }
}

module.exports = GraphAlgorithms;