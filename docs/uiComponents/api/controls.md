# UI Controls API Reference

## Custom Controls

### AgentFlowDiagram

**Module:** `com.sap.a2a.control.AgentFlowDiagram`

**Extends:** `sap.ui.core.Control`

#### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `flowData` | object | {} | Flow diagram data structure |
| `interactive` | boolean | true | Enable interactive features |
| `zoomLevel` | float | 1.0 | Current zoom level |
| `selectedNodeId` | string | "" | Currently selected node |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `zoomIn()` | - | void | Increase zoom level |
| `zoomOut()` | - | void | Decrease zoom level |
| `fitToScreen()` | - | void | Fit diagram to viewport |
| `selectNode(nodeId)` | string | boolean | Select specific node |
| `exportDiagram()` | - | Promise | Export as image |

#### Events

| Event | Parameters | Description |
|-------|------------|-------------|
| `nodeSelect` | {nodeId: string, nodeData: object} | Fired when node is selected |
| `nodeDoubleClick` | {nodeId: string} | Fired on node double-click |
| `connectionSelect` | {connectionId: string} | Fired when connection is selected |
| `diagramChange` | {changeType: string, data: object} | Fired on diagram modification |

#### Example

```javascript
var oDiagram = new AgentFlowDiagram({
    flowData: {
        nodes: [{id: "1", label: "Start"}],
        connections: []
    },
    nodeSelect: function(oEvent) {
        var sNodeId = oEvent.getParameter("nodeId");
        console.log("Selected node:", sNodeId);
    }
});
```

---

### AgentStatusIndicator

**Module:** `com.sap.a2a.control.AgentStatusIndicator`

**Extends:** `sap.ui.core.Control`

#### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `status` | string | "inactive" | Agent status (active/inactive/error/warning) |
| `size` | string | "Medium" | Indicator size (Small/Medium/Large) |
| `animated` | boolean | true | Enable pulse animation |
| `showTooltip` | boolean | true | Show status tooltip |

#### Aggregations

| Aggregation | Type | Cardinality | Description |
|-------------|------|-------------|-------------|
| `tooltip` | sap.ui.core.TooltipBase | 0..1 | Custom tooltip |

#### CSS Classes

- `.agentStatusIndicator` - Base class
- `.status-active` - Active status styling
- `.status-inactive` - Inactive status styling
- `.status-error` - Error status styling
- `.status-warning` - Warning status styling

---

### LoadingIndicator

**Module:** `com.sap.a2a.control.LoadingIndicator`

**Extends:** `sap.ui.core.Control`

#### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `text` | string | "Loading..." | Loading text |
| `type` | string | "circular" | Indicator type (circular/linear/dots) |
| `size` | string | "Medium" | Indicator size |
| `color` | string | "Default" | Color scheme |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `start()` | - | void | Start animation |
| `stop()` | - | void | Stop animation |
| `setProgress(value)` | float | void | Set progress (linear type only) |

---

### NetworkVisualization

**Module:** `com.sap.a2a.control.NetworkVisualization`

**Extends:** `sap.ui.core.Control`

#### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `nodes` | array | [] | Network nodes data |
| `edges` | array | [] | Network connections |
| `layout` | string | "force" | Layout algorithm |
| `physics` | boolean | true | Enable physics simulation |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `addNode(node)` | object | string | Add node to network |
| `removeNode(nodeId)` | string | boolean | Remove node from network |
| `updateLayout()` | - | void | Recalculate layout |
| `getSelectedNodes()` | - | array | Get selected nodes |

---

### MetricsCard

**Module:** `com.sap.a2a.control.MetricsCard`

**Extends:** `sap.f.Card`

#### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `metricType` | string | "numeric" | Metric display type |
| `value` | float | 0 | Current metric value |
| `trend` | string | "neutral" | Trend indicator |
| `sparklineData` | array | [] | Historical data points |

#### Aggregations

| Aggregation | Type | Cardinality | Description |
|-------------|------|-------------|-------------|
| `actions` | sap.m.Button | 0..n | Card actions |
| `kpiHeader` | sap.m.NumericContent | 0..1 | KPI header content |

---

### CodeEditor

**Module:** `com.sap.a2a.control.CodeEditor`

**Extends:** `sap.ui.codeeditor.CodeEditor`

#### Additional Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `enableAutocomplete` | boolean | true | Enable code completion |
| `enableLinting` | boolean | true | Enable real-time linting |
| `theme` | string | "tomorrow" | Editor theme |

#### Additional Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `formatCode()` | - | void | Auto-format code |
| `validateSyntax()` | - | object | Validate code syntax |
| `insertSnippet(snippet)` | string | void | Insert code snippet |

---

## Usage Guidelines

### Performance Considerations

1. **Lazy Loading**: Load controls only when needed
   ```javascript
   sap.ui.require(["com/sap/a2a/control/AgentFlowDiagram"], function(AgentFlowDiagram) {
       // Use control
   });
   ```

2. **Data Binding**: Use model binding for dynamic updates
   ```xml
   <custom:AgentStatusIndicator status="{agent>/status}"/>
   ```

3. **Event Handling**: Detach listeners when not needed
   ```javascript
   onExit: function() {
       this._diagram.detachNodeSelect(this.onNodeSelect);
   }
   ```

### Accessibility

All custom controls must:
- Provide keyboard navigation
- Include ARIA attributes
- Support high contrast themes
- Announce state changes

### Theming

Controls automatically adapt to active theme:
```javascript
// Controls respond to theme changes
sap.ui.getCore().attachThemeChanged(function() {
    // Control updates automatically
});
```