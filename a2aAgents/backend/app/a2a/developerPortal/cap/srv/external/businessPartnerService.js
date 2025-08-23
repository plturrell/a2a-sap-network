"use strict";

/**
 * SAP Cloud SDK Business Partner VDM Service
 * Integrates with SAP S/4HANA Business Partner API
 */

const { BusinessPartner, BusinessPartnerAddress } = require('@sap-cloud-sdk/vdm-business-partner-service');
const { buildHttpRequest } = require('@sap-cloud-sdk/http-client');
const { executeHttpRequest } = require('@sap-cloud-sdk/http-client');
const { getDestination } = require('@sap-cloud-sdk/connectivity');

class BusinessPartnerService {
    constructor() {
        this.destinationName = process.env.S4HANA_DESTINATION || 'S4HANA-PROD';
        this.cache = new Map();
        this.cacheTimeout = 300000; // 5 minutes
    }

    /**
     * Get business partner by ID
     */
    async getBusinessPartner(businessPartnerId) {
        try {
            // Check cache first
            const cached = this._getFromCache(`bp_${businessPartnerId}`);
            if (cached) {
return cached;
}

            const destination = await getDestination(this.destinationName);
            
            const businessPartner = await BusinessPartner
                .requestBuilder()
                .getByKey(businessPartnerId)
                .select(
                    BusinessPartner.BUSINESS_PARTNER,
                    BusinessPartner.CUSTOMER,
                    BusinessPartner.SUPPLIER,
                    BusinessPartner.FIRST_NAME,
                    BusinessPartner.LAST_NAME,
                    BusinessPartner.BUSINESS_PARTNER_CATEGORY,
                    BusinessPartner.BUSINESS_PARTNER_GROUP,
                    BusinessPartner.BUSINESS_PARTNER_NAME,
                    BusinessPartner.ORGANIZATION_BP_NAME_1,
                    BusinessPartner.ORGANIZATION_BP_NAME_2,
                    BusinessPartner.CREATED_BY_USER,
                    BusinessPartner.CREATION_DATE,
                    BusinessPartner.TO_BUSINESS_PARTNER_ADDRESS
                )
                .expand(BusinessPartner.TO_BUSINESS_PARTNER_ADDRESS)
                .execute(destination);

            // Cache the result
            this._setCache(`bp_${businessPartnerId}`, businessPartner);
            
            return this._transformBusinessPartner(businessPartner);
        } catch (error) {
            console.error('Failed to fetch business partner:', error);
            throw new Error(`Business Partner ${businessPartnerId} not found`);
        }
    }

    /**
     * Get all business partners with filters
     */
    async getBusinessPartners(filters = {}) {
        try {
            const destination = await getDestination(this.destinationName);
            
            let requestBuilder = BusinessPartner
                .requestBuilder()
                .getAll()
                .select(
                    BusinessPartner.BUSINESS_PARTNER,
                    BusinessPartner.BUSINESS_PARTNER_NAME,
                    BusinessPartner.BUSINESS_PARTNER_CATEGORY,
                    BusinessPartner.BUSINESS_PARTNER_GROUP,
                    BusinessPartner.CREATED_BY_USER,
                    BusinessPartner.CREATION_DATE
                )
                .top(filters.limit || 100)
                .skip(filters.offset || 0);

            // Apply filters
            if (filters.category) {
                requestBuilder = requestBuilder.filter(
                    BusinessPartner.BUSINESS_PARTNER_CATEGORY.equals(filters.category)
                );
            }

            if (filters.group) {
                requestBuilder = requestBuilder.filter(
                    BusinessPartner.BUSINESS_PARTNER_GROUP.equals(filters.group)
                );
            }

            if (filters.name) {
                requestBuilder = requestBuilder.filter(
                    BusinessPartner.BUSINESS_PARTNER_NAME.contains(filters.name)
                );
            }

            const businessPartners = await requestBuilder.execute(destination);

            return businessPartners.map(bp => this._transformBusinessPartner(bp));
        } catch (error) {
            console.error('Failed to fetch business partners:', error);
            throw error;
        }
    }

    /**
     * Create new business partner
     */
    async createBusinessPartner(businessPartnerData) {
        try {
            const destination = await getDestination(this.destinationName);
            
            const newBusinessPartner = BusinessPartner.builder()
                .businessPartnerCategory(businessPartnerData.category || '1') // Person
                .businessPartnerGroup(businessPartnerData.group || 'BP01')
                .firstName(businessPartnerData.firstName)
                .lastName(businessPartnerData.lastName)
                .organizationBpName1(businessPartnerData.organizationName)
                .build();

            const createdBusinessPartner = await BusinessPartner
                .requestBuilder()
                .create(newBusinessPartner)
                .execute(destination);

            // Clear cache
            this._clearCache();

            return this._transformBusinessPartner(createdBusinessPartner);
        } catch (error) {
            console.error('Failed to create business partner:', error);
            throw error;
        }
    }

    /**
     * Update business partner
     */
    async updateBusinessPartner(businessPartnerId, updateData) {
        try {
            const destination = await getDestination(this.destinationName);
            
            // First get the existing business partner
            const existingBP = await BusinessPartner
                .requestBuilder()
                .getByKey(businessPartnerId)
                .execute(destination);

            // Update fields
            if (updateData.firstName) {
existingBP.firstName = updateData.firstName;
}
            if (updateData.lastName) {
existingBP.lastName = updateData.lastName;
}
            if (updateData.organizationName) {
existingBP.organizationBpName1 = updateData.organizationName;
}

            const updatedBusinessPartner = await BusinessPartner
                .requestBuilder()
                .update(existingBP)
                .execute(destination);

            // Clear cache for this BP
            this._clearCache(`bp_${businessPartnerId}`);

            return this._transformBusinessPartner(updatedBusinessPartner);
        } catch (error) {
            console.error('Failed to update business partner:', error);
            throw error;
        }
    }

    /**
     * Get business partner addresses
     */
    async getBusinessPartnerAddresses(businessPartnerId) {
        try {
            const destination = await getDestination(this.destinationName);
            
            const addresses = await BusinessPartnerAddress
                .requestBuilder()
                .getAll()
                .filter(
                    BusinessPartnerAddress.BUSINESS_PARTNER.equals(businessPartnerId)
                )
                .execute(destination);

            return addresses.map(addr => this._transformAddress(addr));
        } catch (error) {
            console.error('Failed to fetch addresses:', error);
            throw error;
        }
    }

    /**
     * Link business partner to project
     */
    async linkToProject(businessPartnerId, projectId, role = 'CUSTOMER') {
        // This would typically update a custom field or create a relationship
        // For now, we'll simulate this with a custom API call
        try {
            const destination = await getDestination(this.destinationName);
            
            const customRequest = buildHttpRequest({
                method: 'POST',
                url: `/sap/opu/odata/sap/API_BUSINESS_PARTNER/A_BusinessPartnerProject`,
                data: {
                    BusinessPartner: businessPartnerId,
                    ProjectID: projectId,
                    Role: role
                }
            });

            await executeHttpRequest(destination, customRequest);
            
            return {
                businessPartnerId,
                projectId,
                role,
                linkedAt: new Date().toISOString()
            };
        } catch (error) {
            console.error('Failed to link business partner to project:', error);
            throw error;
        }
    }

    /**
     * Transform S/4HANA business partner to internal format
     */
    _transformBusinessPartner(businessPartner) {
        return {
            id: businessPartner.businessPartner,
            type: this._mapBusinessPartnerType(businessPartner.businessPartnerCategory),
            name: businessPartner.businessPartnerName || 
                  `${businessPartner.firstName || ''} ${businessPartner.lastName || ''}`.trim() ||
                  businessPartner.organizationBpName1,
            firstName: businessPartner.firstName,
            lastName: businessPartner.lastName,
            organizationName: businessPartner.organizationBpName1,
            group: businessPartner.businessPartnerGroup,
            isCustomer: businessPartner.customer !== '',
            isSupplier: businessPartner.supplier !== '',
            createdBy: businessPartner.createdByUser,
            createdAt: businessPartner.creationDate,
            addresses: businessPartner.toBusinessPartnerAddress || []
        };
    }

    /**
     * Transform address to internal format
     */
    _transformAddress(address) {
        return {
            id: address.addressId,
            streetName: address.streetName,
            houseNumber: address.houseNumber,
            postalCode: address.postalCode,
            cityName: address.cityName,
            country: address.country,
            region: address.region
        };
    }

    /**
     * Map business partner category to type
     */
    _mapBusinessPartnerType(category) {
        const typeMap = {
            '1': 'PERSON',
            '2': 'ORGANIZATION',
            '3': 'GROUP'
        };
        return typeMap[category] || 'UNKNOWN';
    }

    /**
     * Cache management
     */
    _getFromCache(key) {
        const cached = this.cache.get(key);
        if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
            return cached.data;
        }
        this.cache.delete(key);
        return null;
    }

    _setCache(key, data) {
        this.cache.set(key, {
            data,
            timestamp: Date.now()
        });
    }

    _clearCache(key) {
        if (key) {
            this.cache.delete(key);
        } else {
            this.cache.clear();
        }
    }
}

module.exports = new BusinessPartnerService();