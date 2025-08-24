'use strict';

/**
 * SAP Cloud SDK Service - Enterprise Integration Layer
 * Comprehensive VDM implementation with resilience patterns
 */

const { 
  BusinessPartner,
  SalesOrder,
  Customer: _Customer,
  Supplier: _Supplier,
  Product: _Product,
  Material
} = require('@sap-cloud-sdk/vdm-business-partner-service');

const {
  SalesOrderItem,
  SalesOrderHeaderPartner,
  SalesOrderScheduleLine: _SalesOrderScheduleLine
} = require('@sap-cloud-sdk/vdm-sales-order-service');

const {
  executeHttpRequest,
  and,
  or
} = require('@sap-cloud-sdk/core');

const {
  circuitBreaker: _circuitBreaker,
  retry: _retry,
  timeout,
  resilience
} = require('@sap-cloud-sdk/resilience');

const cacheManager = require('../cache/cache-manager');
const performanceMonitor = require('../monitoring/performance-monitor');
const tracer = require('../telemetry/tracer');

class SAPCloudSDKService {
  constructor() {
    this.destination = null;
    this.initialized = false;
        
    // Configure resilience patterns
    this.resilienceConfig = {
      circuitBreaker: {
        timeout: 10000,
        errorThresholdPercentage: 50,
        requestVolumeThreshold: 10,
        sleepWindowInMilliseconds: 30000
      },
      retry: {
        count: 3,
        delay: 1000,
        backoffMultiplier: 2
      },
      timeout: {
        duration: 30000
      }
    };
  }

  async initialize(destination) {
    if (this.initialized) {
      return;
    }

    const span = tracer.startSpan('sap-cloud-sdk.initialize');
        
    try {
      this.destination = destination;
            
      // Test connection
      await this.healthCheck();
            
      this.initialized = true;
      span.setStatus({ code: 1, message: 'SAP Cloud SDK initialized' });
            
    } catch (error) {
      span.recordException(error);
      span.setStatus({ code: 2, message: error.message });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
     * Business Partner Operations
     */
    
  // Get Business Partner with full entity expansion
  async getBusinessPartner(businessPartnerId) {
    const span = tracer.startSpan('sap-cloud-sdk.get_business_partner');
    span.setAttribute('businessPartner.id', businessPartnerId);
        
    try {
      // Check cache first
      const cacheKey = `bp:full:${businessPartnerId}`;
      const cached = await cacheManager.get(cacheKey, 'businessPartner');
      if (cached) {
        performanceMonitor.recordCacheAccess('businessPartner', 'get', true);
        return cached;
      }

      // Execute with resilience patterns
      const businessPartner = await resilience(
        () => BusinessPartner
          .requestBuilder()
          .getByKey(businessPartnerId)
          .select(
            BusinessPartner.BUSINESS_PARTNER,
            BusinessPartner.CUSTOMER,
            BusinessPartner.SUPPLIER,
            BusinessPartner.BUSINESS_PARTNER_CATEGORY,
            BusinessPartner.BUSINESS_PARTNER_FULL_NAME,
            BusinessPartner.CREATED_BY_USER,
            BusinessPartner.CREATION_DATE,
            BusinessPartner.LAST_CHANGED_BY_USER,
            BusinessPartner.LAST_CHANGE_DATE
          )
          .expand(
            BusinessPartner.TO_BUSINESS_PARTNER_ADDRESS,
            BusinessPartner.TO_BUSINESS_PARTNER_ROLE,
            BusinessPartner.TO_BUSINESS_PARTNER_TAX
          )
          .execute(this.destination),
        this.resilienceConfig
      );

      // Cache the result
      await cacheManager.set(cacheKey, businessPartner, 'businessPartner', 3600);
            
      performanceMonitor.recordDbQuery('sap_s4hana', 'business_partner', 'success');
      span.setAttribute('businessPartner.found', !!businessPartner);
            
      return businessPartner;
            
    } catch (error) {
      performanceMonitor.recordDbQuery('sap_s4hana', 'business_partner', 'error');
      span.recordException(error);
      span.setStatus({ code: 2, message: error.message });
      throw error;
    } finally {
      span.end();
    }
  }

  // Search Business Partners with complex filters
  async searchBusinessPartners(criteria) {
    const span = tracer.startSpan('sap-cloud-sdk.search_business_partners');
        
    try {
      const {
        searchTerm,
        category,
        country: _country,
        city: _city,
        limit = 50,
        skip = 0
      } = criteria;

      // Build complex filter
      const filters = [];
            
      if (searchTerm) {
        filters.push(
          or(
            BusinessPartner.BUSINESS_PARTNER_FULL_NAME.contains(searchTerm),
            BusinessPartner.BUSINESS_PARTNER.contains(searchTerm)
          )
        );
      }
            
      if (category) {
        filters.push(BusinessPartner.BUSINESS_PARTNER_CATEGORY.equals(category));
      }

      // Execute search with resilience
      const businessPartners = await resilience(
        () => BusinessPartner
          .requestBuilder()
          .getAll()
          .filter(and(...filters))
          .select(
            BusinessPartner.BUSINESS_PARTNER,
            BusinessPartner.BUSINESS_PARTNER_FULL_NAME,
            BusinessPartner.BUSINESS_PARTNER_CATEGORY,
            BusinessPartner.CREATED_BY_USER,
            BusinessPartner.CREATION_DATE
          )
          .top(limit)
          .skip(skip)
          .orderBy(BusinessPartner.CREATION_DATE.desc())
          .execute(this.destination),
        this.resilienceConfig
      );

      performanceMonitor.recordDbQuery('sap_s4hana', 'business_partner_search', 'success');
      span.setAttribute('search.results', businessPartners.length);
            
      return businessPartners;
            
    } catch (error) {
      performanceMonitor.recordDbQuery('sap_s4hana', 'business_partner_search', 'error');
      span.recordException(error);
      throw error;
    } finally {
      span.end();
    }
  }

  // Create Business Partner with full validation
  async createBusinessPartner(businessPartnerData) {
    const span = tracer.startSpan('sap-cloud-sdk.create_business_partner');
        
    try {
      // Validate required fields
      this._validateBusinessPartnerData(businessPartnerData);

      // Create with resilience patterns
      const newBusinessPartner = await resilience(
        () => BusinessPartner
          .requestBuilder()
          .create({
            businessPartnerCategory: businessPartnerData.category,
            businessPartnerFullName: businessPartnerData.fullName,
            createdByUser: businessPartnerData.createdBy,
            ...businessPartnerData.additionalFields
          })
          .execute(this.destination),
        this.resilienceConfig
      );

      // Invalidate related caches
      await this._invalidateBusinessPartnerCaches();
            
      performanceMonitor.recordDbQuery('sap_s4hana', 'business_partner_create', 'success');
      span.setAttribute('businessPartner.created', newBusinessPartner.businessPartner);
            
      return newBusinessPartner;
            
    } catch (error) {
      performanceMonitor.recordDbQuery('sap_s4hana', 'business_partner_create', 'error');
      span.recordException(error);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
     * Sales Order Operations
     */
    
  // Get Sales Order with all items and details
  async getSalesOrder(salesOrderId) {
    const span = tracer.startSpan('sap-cloud-sdk.get_sales_order');
    span.setAttribute('salesOrder.id', salesOrderId);
        
    try {
      // Check cache
      const cacheKey = `so:full:${salesOrderId}`;
      const cached = await cacheManager.get(cacheKey, 'salesOrder');
      if (cached) {
        performanceMonitor.recordCacheAccess('salesOrder', 'get', true);
        return cached;
      }

      // Execute with resilience
      const salesOrder = await resilience(
        () => SalesOrder
          .requestBuilder()
          .getByKey(salesOrderId)
          .select(
            SalesOrder.SALES_ORDER,
            SalesOrder.SALES_ORDER_TYPE,
            SalesOrder.SALES_ORGANIZATION,
            SalesOrder.DISTRIBUTION_CHANNEL,
            SalesOrder.ORGANIZATIONAL_DIVISION,
            SalesOrder.SOLD_TO_PARTY,
            SalesOrder.CREATION_DATE,
            SalesOrder.CREATED_BY_USER,
            SalesOrder.LAST_CHANGE_DATE,
            SalesOrder.TOTAL_NET_AMOUNT,
            SalesOrder.TRANSACTION_CURRENCY
          )
          .expand(
            SalesOrder.TO_ITEM.select(
              SalesOrderItem.SALES_ORDER_ITEM,
              SalesOrderItem.MATERIAL,
              SalesOrderItem.REQUESTED_QUANTITY,
              SalesOrderItem.REQUESTED_QUANTITY_UNIT,
              SalesOrderItem.ITEM_GROSS_WEIGHT,
              SalesOrderItem.ITEM_NET_WEIGHT,
              SalesOrderItem.NET_AMOUNT,
              SalesOrderItem.TRANSACTION_CURRENCY
            ),
            SalesOrder.TO_PARTNER.select(
              SalesOrderHeaderPartner.PARTNER_FUNCTION,
              SalesOrderHeaderPartner.CUSTOMER
            )
          )
          .execute(this.destination),
        this.resilienceConfig
      );

      // Cache for 10 minutes (sales orders change frequently)
      await cacheManager.set(cacheKey, salesOrder, 'salesOrder', 600);
            
      performanceMonitor.recordDbQuery('sap_s4hana', 'sales_order', 'success');
      span.setAttribute('salesOrder.found', !!salesOrder);
            
      return salesOrder;
            
    } catch (error) {
      performanceMonitor.recordDbQuery('sap_s4hana', 'sales_order', 'error');
      span.recordException(error);
      throw error;
    } finally {
      span.end();
    }
  }

  // Create Sales Order from A2A Project
  async createSalesOrderFromProject(projectData) {
    const span = tracer.startSpan('sap-cloud-sdk.create_sales_order_from_project');
        
    try {
      const salesOrderData = this._transformProjectToSalesOrder(projectData);
            
      // Create sales order with items
      const salesOrder = await resilience(
        () => SalesOrder
          .requestBuilder()
          .create({
            salesOrderType: salesOrderData.type,
            salesOrganization: salesOrderData.salesOrg,
            distributionChannel: salesOrderData.distributionChannel,
            organizationalDivision: salesOrderData.division,
            soldToParty: salesOrderData.soldToParty,
            pricingDate: new Date().toISOString().split('T')[0],
            requestedDeliveryDate: salesOrderData.requestedDeliveryDate,
                        
            // Add items
            toItem: salesOrderData.items.map(item => ({
              salesOrderItem: item.itemNumber,
              material: item.material,
              requestedQuantity: item.quantity,
              requestedQuantityUnit: item.unit,
              itemCategory: item.category
            }))
          })
          .execute(this.destination),
        this.resilienceConfig
      );

      // Link back to project
      await this._linkSalesOrderToProject(salesOrder.salesOrder, projectData.projectId);
            
      performanceMonitor.recordDbQuery('sap_s4hana', 'sales_order_create', 'success');
      span.setAttribute('salesOrder.created', salesOrder.salesOrder);
            
      return salesOrder;
            
    } catch (error) {
      performanceMonitor.recordDbQuery('sap_s4hana', 'sales_order_create', 'error');
      span.recordException(error);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
     * Product/Material Operations
     */
    
  // Get Product details with pricing
  async getProduct(materialId) {
    const span = tracer.startSpan('sap-cloud-sdk.get_product');
        
    try {
      const cacheKey = `product:${materialId}`;
      const cached = await cacheManager.get(cacheKey, 'product');
      if (cached) {
        return cached;
      }

      const product = await resilience(
        () => Material
          .requestBuilder()
          .getByKey(materialId)
          .select(
            Material.MATERIAL,
            Material.MATERIAL_TYPE,
            Material.MATERIAL_GROUP,
            Material.BASE_UNIT,
            Material.PRODUCT_HIERARCHY,
            Material.DIVISION,
            Material.GROSS_WEIGHT,
            Material.NET_WEIGHT,
            Material.WEIGHT_UNIT
          )
          .execute(this.destination),
        this.resilienceConfig
      );

      // Cache for 24 hours (products don't change often)
      await cacheManager.set(cacheKey, product, 'product', 86400);
            
      return product;
            
    } catch (error) {
      span.recordException(error);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
     * Batch Operations for Performance
     */
    
  // Execute multiple operations in a single batch
  async executeBatch(operations) {
    const span = tracer.startSpan('sap-cloud-sdk.execute_batch');
    span.setAttribute('batch.operations', operations.length);
        
    try {
      const batchResponses = await resilience(
        () => executeHttpRequest(
          this.destination,
          {
            method: 'POST',
            url: '$batch',
            headers: {
              'Content-Type': 'multipart/mixed'
            },
            data: this._buildBatchRequest(operations)
          }
        ),
        this.resilienceConfig
      );

      performanceMonitor.recordDbQuery('sap_s4hana', 'batch_operation', 'success');
      span.setAttribute('batch.success', true);
            
      return this._parseBatchResponse(batchResponses);
            
    } catch (error) {
      performanceMonitor.recordDbQuery('sap_s4hana', 'batch_operation', 'error');
      span.recordException(error);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
     * Health Check and Diagnostics
     */
    
  async healthCheck() {
    const span = tracer.startSpan('sap-cloud-sdk.health_check');
        
    try {
      // Test basic connectivity
      await timeout(
        () => BusinessPartner
          .requestBuilder()
          .getAll()
          .top(1)
          .execute(this.destination),
        5000 // 5 second timeout
      );
            
      span.setAttribute('health.status', 'UP');
      return { status: 'UP', timestamp: new Date().toISOString() };
            
    } catch (error) {
      span.recordException(error);
      span.setAttribute('health.status', 'DOWN');
      return { 
        status: 'DOWN', 
        error: error.message,
        timestamp: new Date().toISOString() 
      };
    } finally {
      span.end();
    }
  }

  /**
     * Private Helper Methods
     */
    
  _validateBusinessPartnerData(data) {
    const required = ['category', 'fullName'];
    const missing = required.filter(field => !data[field]);
        
    if (missing.length > 0) {
      throw new Error(`Missing required fields: ${missing.join(', ')}`);
    }
  }

  _transformProjectToSalesOrder(projectData) {
    return {
      type: 'OR', // Standard order
      salesOrg: '1000',
      distributionChannel: '10',
      division: '00',
      soldToParty: projectData.businessPartnerId,
      requestedDeliveryDate: projectData.targetDate || 
                new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      items: projectData.services?.map((service, index) => ({
        itemNumber: (index + 1) * 10,
        material: service.materialCode || 'SERVICE-001',
        quantity: service.quantity || 1,
        unit: 'EA',
        category: 'TAN'
      })) || []
    };
  }

  async _linkSalesOrderToProject(salesOrderId, projectId) {
    await cacheManager.set(
      `project:so:${projectId}`,
      { salesOrderId, linkedAt: new Date().toISOString() },
      'project'
    );
  }

  async _invalidateBusinessPartnerCaches() {
    await Promise.all([
      cacheManager.clearPattern('bp:*', 'businessPartner'),
      cacheManager.clearPattern('bp:search:*', 'businessPartner')
    ]);
  }

  _buildBatchRequest(operations) {
    // Implementation for batch request building
    return operations.map(op => op.request).join('\n');
  }

  _parseBatchResponse(response) {
    // Implementation for batch response parsing
    return response.data;
  }
}

// Export singleton instance
module.exports = new SAPCloudSDKService();