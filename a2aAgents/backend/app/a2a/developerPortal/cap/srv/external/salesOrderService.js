'use strict';

/**
 * SAP Cloud SDK Sales Order VDM Service
 * Integrates with SAP S/4HANA Sales Order API
 */

const { 
  SalesOrder, 
  SalesOrderItem,
  SalesOrderScheduleLine: _SalesOrderScheduleLine 
} = require('@sap-cloud-sdk/vdm-sales-order-service');
const { getDestination } = require('@sap-cloud-sdk/connectivity');
const performanceMonitor = require('../monitoring/performance-monitor');

class SalesOrderService {
  constructor() {
    this.destinationName = process.env.S4HANA_DESTINATION || 'S4HANA-PROD';
  }

  /**
     * Create sales order from project deployment
     */
  async createSalesOrderFromProject(project) {
    const startMark = `sales-order-create-${Date.now()}`;
    performanceMonitor.mark(startMark);

    try {
      const destination = await getDestination(this.destinationName);
            
      // Build sales order from project data
      const salesOrderData = this._buildSalesOrderData(project);
            
      const newSalesOrder = SalesOrder.builder()
        .salesOrderType(salesOrderData.orderType)
        .salesOrganization(salesOrderData.salesOrg)
        .distributionChannel(salesOrderData.distributionChannel)
        .organizationDivision(salesOrderData.division)
        .soldToParty(salesOrderData.customerId)
        .purchaseOrderByCustomer(project.projectId)
        .requestedDeliveryDate(salesOrderData.deliveryDate)
        .build();

      // Create sales order
      const createdOrder = await SalesOrder
        .requestBuilder()
        .create(newSalesOrder)
        .execute(destination);

      // Add items
      const items = await this._createSalesOrderItems(
        createdOrder.salesOrder,
        project,
        destination
      );

      const endMark = `sales-order-create-end-${Date.now()}`;
      performanceMonitor.mark(endMark);
      performanceMonitor.measure('sales-order-creation', startMark, endMark);

      return {
        salesOrderNumber: createdOrder.salesOrder,
        totalAmount: createdOrder.totalNetAmount,
        currency: createdOrder.transactionCurrency,
        items: items.length,
        status: createdOrder.overallDeliveryStatus,
        createdAt: new Date().toISOString()
      };
    } catch (error) {
      console.error('Failed to create sales order:', error);
      performanceMonitor.recordError('sales_order_creation', 'sales-order-service');
      throw error;
    }
  }

  /**
     * Get sales order details
     */
  async getSalesOrder(salesOrderNumber) {
    try {
      const destination = await getDestination(this.destinationName);
            
      const salesOrder = await SalesOrder
        .requestBuilder()
        .getByKey(salesOrderNumber)
        .select(
          SalesOrder.SALES_ORDER,
          SalesOrder.SALES_ORDER_TYPE,
          SalesOrder.SALES_ORGANIZATION,
          SalesOrder.SOLD_TO_PARTY,
          SalesOrder.TOTAL_NET_AMOUNT,
          SalesOrder.TRANSACTION_CURRENCY,
          SalesOrder.OVERALL_DELIVERY_STATUS,
          SalesOrder.OVERALL_SD_PROCESS_STATUS,
          SalesOrder.CREATION_DATE,
          SalesOrder.TO_ITEM,
          SalesOrder.TO_PARTNER
        )
        .expand(
          SalesOrder.TO_ITEM.expand(
            SalesOrderItem.TO_SCHEDULE_LINE
          )
        )
        .execute(destination);

      return this._transformSalesOrder(salesOrder);
    } catch (error) {
      console.error('Failed to fetch sales order:', error);
      throw error;
    }
  }

  /**
     * Get sales orders for a project
     */
  async getSalesOrdersForProject(projectId) {
    try {
      const destination = await getDestination(this.destinationName);
            
      const salesOrders = await SalesOrder
        .requestBuilder()
        .getAll()
        .filter(
          SalesOrder.PURCHASE_ORDER_BY_CUSTOMER.equals(projectId)
        )
        .select(
          SalesOrder.SALES_ORDER,
          SalesOrder.SALES_ORDER_TYPE,
          SalesOrder.TOTAL_NET_AMOUNT,
          SalesOrder.TRANSACTION_CURRENCY,
          SalesOrder.OVERALL_DELIVERY_STATUS,
          SalesOrder.CREATION_DATE
        )
        .orderBy(SalesOrder.CREATION_DATE.desc())
        .top(50)
        .execute(destination);

      return salesOrders.map(so => this._transformSalesOrderSummary(so));
    } catch (error) {
      console.error('Failed to fetch sales orders for project:', error);
      throw error;
    }
  }

  /**
     * Update sales order status
     */
  async updateSalesOrderStatus(salesOrderNumber, newStatus) {
    try {
      const destination = await getDestination(this.destinationName);
            
      // Get existing order
      const existingOrder = await SalesOrder
        .requestBuilder()
        .getByKey(salesOrderNumber)
        .execute(destination);

      // Update status fields based on business logic
      if (newStatus === 'CANCELLED') {
        existingOrder.overallSDProcessStatus = 'C'; // Cancelled
      } else if (newStatus === 'COMPLETED') {
        existingOrder.overallDeliveryStatus = 'C'; // Completely delivered
      }

      const updatedOrder = await SalesOrder
        .requestBuilder()
        .update(existingOrder)
        .execute(destination);

      return this._transformSalesOrder(updatedOrder);
    } catch (error) {
      console.error('Failed to update sales order:', error);
      throw error;
    }
  }

  /**
     * Create sales order items from project agents
     */
  async _createSalesOrderItems(salesOrderNumber, project, destination) {
    const items = [];
    let itemNumber = 10;

    for (const agent of project.agents) {
      const item = await this._createItemForAgent(
        salesOrderNumber,
        itemNumber,
        agent,
        destination
      );
      items.push(item);
      itemNumber += 10;
    }

    return items;
  }

  /**
     * Create individual sales order item
     */
  async _createItemForAgent(salesOrderNumber, itemNumber, agent, destination) {
    try {
      const newItem = SalesOrderItem.builder()
        .salesOrder(salesOrderNumber)
        .salesOrderItem(itemNumber.toString())
        .material(this._mapAgentToMaterial(agent.type))
        .requestedQuantity(1)
        .requestedQuantityUnit('EA')
        .itemCategory('TAN') // Standard item
        .build();

      const createdItem = await SalesOrderItem
        .requestBuilder()
        .create(newItem)
        .execute(destination);

      return {
        itemNumber: createdItem.salesOrderItem,
        material: createdItem.material,
        quantity: createdItem.requestedQuantity,
        netAmount: createdItem.netAmount
      };
    } catch (error) {
      console.error('Failed to create sales order item:', error);
      throw error;
    }
  }

  /**
     * Build sales order data from project
     */
  _buildSalesOrderData(project) {
    return {
      orderType: 'OR', // Standard Order
      salesOrg: process.env.SALES_ORG || '1000',
      distributionChannel: process.env.DISTRIBUTION_CHANNEL || '10',
      division: process.env.DIVISION || '00',
      customerId: project.businessPartnerId || '0010000000',
      deliveryDate: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000) // 7 days from now
    };
  }

  /**
     * Map agent type to material number
     */
  _mapAgentToMaterial(agentType) {
    const materialMap = {
      'reactive': 'A2A-AGENT-001',
      'proactive': 'A2A-AGENT-002',
      'collaborative': 'A2A-AGENT-003'
    };
    return materialMap[agentType] || 'A2A-AGENT-000';
  }

  /**
     * Transform sales order to internal format
     */
  _transformSalesOrder(salesOrder) {
    return {
      salesOrderNumber: salesOrder.salesOrder,
      type: salesOrder.salesOrderType,
      customer: {
        id: salesOrder.soldToParty,
        name: salesOrder.toPartner?.find(p => p.partnerFunction === 'AG')?.customer
      },
      totalAmount: parseFloat(salesOrder.totalNetAmount),
      currency: salesOrder.transactionCurrency,
      status: {
        delivery: this._mapDeliveryStatus(salesOrder.overallDeliveryStatus),
        process: this._mapProcessStatus(salesOrder.overallSDProcessStatus)
      },
      createdAt: salesOrder.creationDate,
      items: salesOrder.toItem?.map(item => ({
        itemNumber: item.salesOrderItem,
        material: item.material,
        description: item.materialName,
        quantity: parseFloat(item.requestedQuantity),
        unit: item.requestedQuantityUnit,
        netAmount: parseFloat(item.netAmount),
        scheduleLines: item.toScheduleLine?.map(sl => ({
          scheduleLine: sl.scheduleLine,
          deliveryDate: sl.deliveryDate,
          confirmedQuantity: parseFloat(sl.confirmedQuantity)
        }))
      })) || []
    };
  }

  /**
     * Transform sales order summary
     */
  _transformSalesOrderSummary(salesOrder) {
    return {
      salesOrderNumber: salesOrder.salesOrder,
      type: salesOrder.salesOrderType,
      totalAmount: parseFloat(salesOrder.totalNetAmount),
      currency: salesOrder.transactionCurrency,
      deliveryStatus: this._mapDeliveryStatus(salesOrder.overallDeliveryStatus),
      createdAt: salesOrder.creationDate
    };
  }

  /**
     * Map delivery status codes
     */
  _mapDeliveryStatus(status) {
    const statusMap = {
      '': 'NOT_DELIVERED',
      'A': 'PARTIALLY_DELIVERED',
      'B': 'COMPLETELY_DELIVERED',
      'C': 'NOT_RELEVANT'
    };
    return statusMap[status] || 'UNKNOWN';
  }

  /**
     * Map process status codes
     */
  _mapProcessStatus(status) {
    const statusMap = {
      '': 'OPEN',
      'A': 'IN_PROCESS',
      'B': 'COMPLETED',
      'C': 'CANCELLED'
    };
    return statusMap[status] || 'UNKNOWN';
  }
}

module.exports = new SalesOrderService();