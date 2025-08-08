# ADR-002: SAP UI5 as Frontend Framework

## Status
Accepted

## Context
The A2A Agent Portal requires a frontend framework that:
- Provides enterprise-grade UI components
- Follows SAP Fiori design guidelines
- Supports complex business applications
- Offers excellent accessibility and i18n
- Integrates seamlessly with SAP backend services

## Decision
We will use SAP UI5 (version 1.120.0 LTS) with:
- TypeScript for type safety
- Fiori 3 design system (sap_horizon theme)
- Fiori Elements for standardized UI patterns
- UI5 Tooling for modern development workflow

## Consequences

### Positive
- Native SAP look and feel
- Comprehensive enterprise UI components
- Built-in accessibility (WCAG 2.1 AA)
- Excellent internationalization support
- Direct integration with SAP services
- Long-term support from SAP

### Negative
- Larger bundle size compared to React/Vue
- Smaller developer community
- Less flexibility for custom designs
- Steeper learning curve

### Mitigation
- Use UI5 Tooling for optimization
- Implement code splitting
- Provide comprehensive developer documentation
- Leverage Fiori Elements to reduce custom code

## Implementation
```javascript
// UI5 Bootstrap Configuration
data-sap-ui-theme="sap_horizon"
data-sap-ui-libs="sap.m,sap.f,sap.tnt"
data-sap-ui-compatVersion="edge"
data-sap-ui-async="true"
```

## References
- [UI5 Documentation](https://ui5.sap.com)
- [Fiori Design Guidelines](https://experience.sap.com/fiori-design-web/)
- [UI5 TypeScript Guide](https://sap.github.io/ui5-typescript/)