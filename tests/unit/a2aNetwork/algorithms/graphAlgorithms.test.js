const { expect } = require('chai');
const sinon = require('sinon');
const GraphAlgorithms = require('../../../srv/algorithms/graphAlgorithms');

/**
 * Unit tests for Graph Algorithms implementation
 * Following SAP testing standards and enterprise patterns
 */
describe('GraphAlgorithms', () => {
    let graphAlgorithms;
    let sandbox;

    beforeEach(() => {
        sandbox = sinon.createSandbox();
        graphAlgorithms = new GraphAlgorithms();
    });

    afterEach(() => {
        sandbox.restore();
    });

    describe('Graph Creation and Management', () => {
        it('should create an empty graph with correct structure', () => {
            const graph = graphAlgorithms.createGraph();
            
            expect(graph).to.have.property('nodes');
            expect(graph).to.have.property('edges');
            expect(graph).to.have.property('weights');
            expect(graph).to.have.property('metadata');
            
            expect(graph.nodes).to.be.instanceOf(Set);
            expect(graph.edges).to.be.instanceOf(Map);
            expect(graph.weights).to.be.instanceOf(Map);
            expect(graph.metadata).to.be.instanceOf(Map);
            
            expect(graph.nodes.size).to.equal(0);
            expect(graph.edges.size).to.equal(0);
        });

        it('should add nodes with metadata correctly', () => {
            const graph = graphAlgorithms.createGraph();
            const nodeId = 'test-node';
            const metadata = { type: 'file', path: '/test/path' };
            
            graphAlgorithms.addNode(graph, nodeId, metadata);
            
            expect(graph.nodes.has(nodeId)).to.be.true;
            expect(graph.edges.has(nodeId)).to.be.true;
            expect(graph.metadata.has(nodeId)).to.be.true;
            
            const storedMetadata = graph.metadata.get(nodeId);
            expect(storedMetadata).to.include(metadata);
            expect(storedMetadata).to.have.property('addedAt');
        });

        it('should throw error for invalid node ID', () => {
            const graph = graphAlgorithms.createGraph();
            
            expect(() => {
                graphAlgorithms.addNode(graph, null);
            }).to.throw('Node ID is required');
            
            expect(() => {
                graphAlgorithms.addNode(graph, '');
            }).to.throw('Node ID is required');
        });

        it('should add edges with weights correctly', () => {
            const graph = graphAlgorithms.createGraph();
            const fromNode = 'node-a';
            const toNode = 'node-b';
            const weight = 5;
            
            graphAlgorithms.addEdge(graph, fromNode, toNode, weight);
            
            expect(graph.nodes.has(fromNode)).to.be.true;
            expect(graph.nodes.has(toNode)).to.be.true;
            expect(graph.edges.get(fromNode).has(toNode)).to.be.true;
            expect(graph.weights.get(fromNode).get(toNode)).to.equal(weight);
        });
    });

    describe('Depth-First Search (DFS)', () => {
        let testGraph;

        beforeEach(() => {
            // Create a test graph: A -> B -> D, A -> C -> D
            testGraph = graphAlgorithms.createGraph();
            graphAlgorithms.addEdge(testGraph, 'A', 'B');
            graphAlgorithms.addEdge(testGraph, 'A', 'C');
            graphAlgorithms.addEdge(testGraph, 'B', 'D');
            graphAlgorithms.addEdge(testGraph, 'C', 'D');
        });

        it('should perform DFS traversal correctly', () => {
            const result = graphAlgorithms.depthFirstSearch(testGraph, 'A');
            
            expect(result).to.have.property('visitOrder');
            expect(result).to.have.property('discovered');
            expect(result).to.have.property('finished');
            expect(result).to.have.property('predecessors');
            expect(result).to.have.property('hasPath');
            
            expect(result.visitOrder[0]).to.equal('A');
            expect(result.visitOrder).to.include('B');
            expect(result.visitOrder).to.include('C');
            expect(result.visitOrder).to.include('D');
            
            expect(result.hasPath('D')).to.be.true;
            expect(result.hasPath('NonExistent')).to.be.false;
        });

        it('should handle disconnected components', () => {
            graphAlgorithms.addNode(testGraph, 'E');
            
            const result = graphAlgorithms.depthFirstSearch(testGraph, 'A');
            
            expect(result.visitOrder).to.include('E');
            expect(result.hasPath('E')).to.be.true;
        });

        it('should track discovery and finish times correctly', () => {
            const result = graphAlgorithms.depthFirstSearch(testGraph, 'A');
            
            expect(result.discovered.get('A')).to.be.lessThan(result.finished.get('A'));
            expect(result.discovered.get('B')).to.be.greaterThan(result.discovered.get('A'));
        });
    });

    describe('Breadth-First Search (BFS)', () => {
        let testGraph;

        beforeEach(() => {
            // Create a test graph: A -> B, A -> C, B -> D, C -> D
            testGraph = graphAlgorithms.createGraph();
            graphAlgorithms.addEdge(testGraph, 'A', 'B');
            graphAlgorithms.addEdge(testGraph, 'A', 'C');
            graphAlgorithms.addEdge(testGraph, 'B', 'D');
            graphAlgorithms.addEdge(testGraph, 'C', 'D');
        });

        it('should perform BFS traversal correctly', () => {
            const result = graphAlgorithms.breadthFirstSearch(testGraph, 'A');
            
            expect(result).to.have.property('visitOrder');
            expect(result).to.have.property('distances');
            expect(result).to.have.property('predecessors');
            expect(result).to.have.property('hasPath');
            expect(result).to.have.property('getPath');
            
            expect(result.visitOrder[0]).to.equal('A');
            expect(result.distances.get('A')).to.equal(0);
            expect(result.distances.get('D')).to.equal(2);
        });

        it('should calculate shortest paths correctly', () => {
            const result = graphAlgorithms.breadthFirstSearch(testGraph, 'A');
            
            const pathToD = result.getPath('D');
            expect(pathToD).to.have.lengthOf(3); // A -> B/C -> D
            expect(pathToD[0]).to.equal('A');
            expect(pathToD[2]).to.equal('D');
        });

        it('should handle unreachable nodes', () => {
            const result = graphAlgorithms.breadthFirstSearch(testGraph, 'NonExistent');
            
            expect(result.visitOrder).to.be.empty;
            expect(result.hasPath('A')).to.be.false;
        });
    });

    describe('Topological Sort', () => {
        let dagGraph;

        beforeEach(() => {
            // Create a DAG: A -> B -> D, A -> C -> D
            dagGraph = graphAlgorithms.createGraph();
            graphAlgorithms.addEdge(dagGraph, 'A', 'B');
            graphAlgorithms.addEdge(dagGraph, 'A', 'C');
            graphAlgorithms.addEdge(dagGraph, 'B', 'D');
            graphAlgorithms.addEdge(dagGraph, 'C', 'D');
        });

        it('should produce valid topological ordering', () => {
            const result = graphAlgorithms.topologicalSort(dagGraph);
            
            expect(result).to.be.an('array');
            expect(result).to.include('A');
            expect(result).to.include('B');
            expect(result).to.include('C');
            expect(result).to.include('D');
            
            const indexA = result.indexOf('A');
            const indexB = result.indexOf('B');
            const indexC = result.indexOf('C');
            const indexD = result.indexOf('D');
            
            expect(indexA).to.be.lessThan(indexB);
            expect(indexA).to.be.lessThan(indexC);
            expect(indexB).to.be.lessThan(indexD);
            expect(indexC).to.be.lessThan(indexD);
        });

        it('should detect cycles and throw error', () => {
            // Add cycle: D -> A
            graphAlgorithms.addEdge(dagGraph, 'D', 'A');
            
            expect(() => {
                graphAlgorithms.topologicalSort(dagGraph);
            }).to.throw('Graph contains a cycle - topological sort not possible');
        });
    });

    describe('Dijkstra\'s Algorithm', () => {
        let weightedGraph;

        beforeEach(() => {
            // Create weighted graph
            weightedGraph = graphAlgorithms.createGraph();
            graphAlgorithms.addEdge(weightedGraph, 'A', 'B', 4);
            graphAlgorithms.addEdge(weightedGraph, 'A', 'C', 2);
            graphAlgorithms.addEdge(weightedGraph, 'B', 'D', 3);
            graphAlgorithms.addEdge(weightedGraph, 'C', 'D', 1);
            graphAlgorithms.addEdge(weightedGraph, 'C', 'E', 5);
            graphAlgorithms.addEdge(weightedGraph, 'D', 'E', 2);
        });

        it('should find shortest paths correctly', () => {
            const result = graphAlgorithms.dijkstra(weightedGraph, 'A');
            
            expect(result).to.have.property('distances');
            expect(result).to.have.property('predecessors');
            expect(result).to.have.property('hasPath');
            expect(result).to.have.property('getPath');
            expect(result).to.have.property('getDistance');
            
            expect(result.getDistance('A')).to.equal(0);
            expect(result.getDistance('C')).to.equal(2);
            expect(result.getDistance('D')).to.equal(3); // A -> C -> D
            expect(result.getDistance('E')).to.equal(5); // A -> C -> D -> E
        });

        it('should construct correct shortest paths', () => {
            const result = graphAlgorithms.dijkstra(weightedGraph, 'A');
            
            const pathToE = result.getPath('E');
            expect(pathToE).to.deep.equal(['A', 'C', 'D', 'E']);
            
            const pathToD = result.getPath('D');
            expect(pathToD).to.deep.equal(['A', 'C', 'D']);
        });

        it('should handle unreachable nodes', () => {
            graphAlgorithms.addNode(weightedGraph, 'F');
            
            const result = graphAlgorithms.dijkstra(weightedGraph, 'A');
            
            expect(result.hasPath('F')).to.be.false;
            expect(result.getDistance('F')).to.equal(Infinity);
        });
    });

    describe('Bellman-Ford Algorithm', () => {
        let graphWithNegative;

        beforeEach(() => {
            graphWithNegative = graphAlgorithms.createGraph();
            graphAlgorithms.addEdge(graphWithNegative, 'A', 'B', 1);
            graphAlgorithms.addEdge(graphWithNegative, 'A', 'C', 4);
            graphAlgorithms.addEdge(graphWithNegative, 'B', 'C', -3);
            graphAlgorithms.addEdge(graphWithNegative, 'B', 'D', 2);
            graphAlgorithms.addEdge(graphWithNegative, 'C', 'D', 3);
        });

        it('should handle negative weights correctly', () => {
            const result = graphAlgorithms.bellmanFord(graphWithNegative, 'A');
            
            expect(result).to.have.property('hasNegativeCycle');
            expect(result.hasNegativeCycle).to.be.false;
            
            expect(result.getDistance('C')).to.equal(-2); // A -> B -> C
            expect(result.getDistance('D')).to.equal(1); // A -> B -> C -> D
        });

        it('should detect negative cycles', () => {
            // Add negative cycle
            graphAlgorithms.addEdge(graphWithNegative, 'D', 'B', -6);
            
            const result = graphAlgorithms.bellmanFord(graphWithNegative, 'A');
            
            expect(result.hasNegativeCycle).to.be.true;
        });
    });

    describe('Strongly Connected Components', () => {
        let directedGraph;

        beforeEach(() => {
            directedGraph = graphAlgorithms.createGraph();
            // Create SCCs: {A, B, C} and {D}
            graphAlgorithms.addEdge(directedGraph, 'A', 'B');
            graphAlgorithms.addEdge(directedGraph, 'B', 'C');
            graphAlgorithms.addEdge(directedGraph, 'C', 'A');
            graphAlgorithms.addEdge(directedGraph, 'B', 'D');
        });

        it('should find strongly connected components correctly', () => {
            const components = graphAlgorithms.findStronglyConnectedComponents(directedGraph);
            
            expect(components).to.be.an('array');
            expect(components).to.have.lengthOf(2);
            
            // Find the component containing A, B, C
            const largeComponent = components.find(comp => comp.includes('A'));
            expect(largeComponent).to.include.members(['A', 'B', 'C']);
            
            // Find the component containing D
            const smallComponent = components.find(comp => comp.includes('D'));
            expect(smallComponent).to.deep.equal(['D']);
        });
    });

    describe('Cycle Detection', () => {
        let cyclicGraph;

        beforeEach(() => {
            cyclicGraph = graphAlgorithms.createGraph();
            graphAlgorithms.addEdge(cyclicGraph, 'A', 'B');
            graphAlgorithms.addEdge(cyclicGraph, 'B', 'C');
            graphAlgorithms.addEdge(cyclicGraph, 'C', 'A'); // Creates cycle
            graphAlgorithms.addEdge(cyclicGraph, 'B', 'D');
        });

        it('should detect cycles correctly', () => {
            const cycles = graphAlgorithms.detectCycles(cyclicGraph);
            
            expect(cycles).to.be.an('array');
            expect(cycles).to.have.lengthOf.greaterThan(0);
            
            const cycle = cycles[0];
            expect(cycle).to.include('A');
            expect(cycle).to.include('B');
            expect(cycle).to.include('C');
        });

        it('should return empty array for acyclic graph', () => {
            const acyclicGraph = graphAlgorithms.createGraph();
            graphAlgorithms.addEdge(acyclicGraph, 'A', 'B');
            graphAlgorithms.addEdge(acyclicGraph, 'A', 'C');
            graphAlgorithms.addEdge(acyclicGraph, 'B', 'D');
            
            const cycles = graphAlgorithms.detectCycles(acyclicGraph);
            
            expect(cycles).to.be.an('array');
            expect(cycles).to.be.empty;
        });
    });

    describe('Error Handling and Edge Cases', () => {
        it('should handle empty graphs gracefully', () => {
            const emptyGraph = graphAlgorithms.createGraph();
            
            const dfsResult = graphAlgorithms.depthFirstSearch(emptyGraph, 'NonExistent');
            expect(dfsResult.visitOrder).to.be.empty;
            
            const bfsResult = graphAlgorithms.breadthFirstSearch(emptyGraph, 'NonExistent');
            expect(bfsResult.visitOrder).to.be.empty;
            
            const topSort = graphAlgorithms.topologicalSort(emptyGraph);
            expect(topSort).to.be.empty;
        });

        it('should handle single node graphs', () => {
            const singleNodeGraph = graphAlgorithms.createGraph();
            graphAlgorithms.addNode(singleNodeGraph, 'Solo');
            
            const dfsResult = graphAlgorithms.depthFirstSearch(singleNodeGraph, 'Solo');
            expect(dfsResult.visitOrder).to.deep.equal(['Solo']);
            
            const bfsResult = graphAlgorithms.breadthFirstSearch(singleNodeGraph, 'Solo');
            expect(bfsResult.visitOrder).to.deep.equal(['Solo']);
        });

        it('should maintain graph immutability in operations', () => {
            const originalGraph = graphAlgorithms.createGraph();
            graphAlgorithms.addEdge(originalGraph, 'A', 'B');
            
            const originalSize = originalGraph.nodes.size;
            
            // Operations should not modify the original graph
            graphAlgorithms.depthFirstSearch(originalGraph, 'A');
            graphAlgorithms.breadthFirstSearch(originalGraph, 'A');
            
            expect(originalGraph.nodes.size).to.equal(originalSize);
        });
    });

    describe('Performance and Scalability', () => {
        it('should handle moderately large graphs efficiently', () => {
            const largeGraph = graphAlgorithms.createGraph();
            
            // Create a graph with 100 nodes
            for (let i = 0; i < 100; i++) {
                for (let j = i + 1; j < Math.min(i + 5, 100); j++) {
                    graphAlgorithms.addEdge(largeGraph, `node-${i}`, `node-${j}`);
                }
            }
            
            const startTime = Date.now();
            const result = graphAlgorithms.dijkstra(largeGraph, 'node-0');
            const endTime = Date.now();
            
            expect(endTime - startTime).to.be.lessThan(1000); // Should complete within 1 second
            expect(result.hasPath('node-99')).to.be.true;
        });
    });

    describe('Integration with SAP Standards', () => {
        it('should follow SAP naming conventions', () => {
            const graph = graphAlgorithms.createGraph();
            
            // Test SAP-style entity names
            graphAlgorithms.addNode(graph, 'BusinessPartner', { entity: 'BusinessPartner' });
            graphAlgorithms.addNode(graph, 'SalesOrder', { entity: 'SalesOrder' });
            graphAlgorithms.addEdge(graph, 'BusinessPartner', 'SalesOrder');
            
            expect(graph.nodes.has('BusinessPartner')).to.be.true;
            expect(graph.nodes.has('SalesOrder')).to.be.true;
        });

        it('should support CAP-style relationships', () => {
            const graph = graphAlgorithms.createGraph();
            
            // Test CAP association patterns
            graphAlgorithms.addEdge(graph, 'Customer', 'Orders', 1);
            graphAlgorithms.addEdge(graph, 'Orders', 'OrderItems', 1);
            
            const result = graphAlgorithms.breadthFirstSearch(graph, 'Customer');
            expect(result.hasPath('OrderItems')).to.be.true;
        });
    });
});