// News Search Integration Standardization Framework
// Bridge your 5D business taxonomy with news search capabilities

class NewsSearchStandardizer {
  constructor(config = {}) {
    // Initialize logger
    this.logger = config.logger || this.createDefaultLogger();
    
    // Load configuration with defaults
    this.config = this.loadConfiguration(config);
    
    // Entity recognition mappings for news search
    this.entityMappings = this.config.entityMappings || {
      // Location entity variations for news search
      locationEntities: {
        // Your standardized → News search variations
        'Singapore': ['Singapore', 'Republic of Singapore', 'SG', 'Lion City'],
        'United States': ['United States', 'USA', 'US', 'America', 'United States of America'],
        'United Kingdom': ['United Kingdom', 'UK', 'Britain', 'Great Britain', 'England'],
        'United Arab Emirates': ['UAE', 'United Arab Emirates', 'Emirates'],
        'China': ['China', 'PRC', 'People\'s Republic of China', 'mainland China'],
        'Hong Kong': ['Hong Kong', 'HK', 'Hong Kong SAR'],
        'Americas': ['Americas', 'North America', 'South America', 'Latin America'],
        'Asia': ['Asia', 'Asia Pacific', 'APAC', 'Asian markets'],
        'Europe': ['Europe', 'European Union', 'EU', 'European markets'],
        'Africa': ['Africa', 'African continent', 'Sub-Saharan Africa']
      },

      // Product/Business line entities for news search
      productEntities: {
        'Investment Banking': ['investment banking', 'M&A', 'mergers and acquisitions', 'capital markets', 'IPO', 'underwriting'],
        'Mergers and Acquisitions': ['M&A', 'mergers', 'acquisitions', 'takeover', 'buyout', 'deal'],
        'Capital Markets': ['capital markets', 'equity markets', 'debt markets', 'bond markets', 'securities'],
        'Foreign Exchange': ['forex', 'FX', 'foreign exchange', 'currency trading', 'FX markets'],
        'Credit Trading': ['credit trading', 'credit markets', 'corporate bonds', 'credit derivatives'],
        'Securities Services': ['securities services', 'custody', 'clearing', 'settlement'],
        'Transaction Banking': ['transaction banking', 'cash management', 'trade finance'],
        'Environmental, Social, and Governance': ['ESG', 'sustainability', 'environmental finance', 'green finance']
      },

      // Account/Financial entities for news search
      accountEntities: {
        'Expected Credit Loss': ['credit loss', 'ECL', 'loan loss', 'provisions', 'impairment'],
        'Risk Weighted Assets': ['RWA', 'capital ratios', 'Basel III', 'capital adequacy'],
        'Net Interest Income': ['net interest income', 'NII', 'interest margin', 'spread income'],
        'Trading Income': ['trading revenue', 'market making', 'proprietary trading'],
        'Fee Income': ['fee income', 'commission income', 'service charges']
      },

      // Industry/Sector entities
      industryEntities: {
        'Banking': ['banking', 'banks', 'financial services', 'commercial banking'],
        'Financial Services': ['financial services', 'fintech', 'financial technology'],
        'Capital Markets': ['capital markets', 'investment banking', 'securities']
      }
    };

    // News categories mapping to your business taxonomy
    this.newsCategories = this.config.newsCategories || {
      // Map news categories to your dimensions
      'Business': ['accounts', 'products', 'measures'],
      'Finance': ['accounts', 'products', 'measures'],
      'Economics': ['locations', 'accounts', 'measures'],
      'Markets': ['products', 'accounts', 'measures'],
      'Technology': ['products'],
      'Politics': ['locations'],
      'Regulation': ['accounts', 'books', 'measures'],
      'ESG': ['products', 'locations'],
      'Risk Management': ['accounts', 'measures'],
      'Mergers & Acquisitions': ['products', 'books']
    };

    // Sentiment impact mapping
    this.sentimentImpact = this.config.sentimentImpact || {
      'positive': {
        keywords: ['growth', 'profit', 'success', 'expansion', 'bullish', 'upgrade', 'outperform'],
        businessImpact: 'favorable',
        priority: 'high'
      },
      'negative': {
        keywords: ['loss', 'decline', 'crisis', 'downgrade', 'bearish', 'risk', 'concern'],
        businessImpact: 'adverse',
        priority: 'critical'
      },
      'neutral': {
        keywords: ['announcement', 'update', 'report', 'statement'],
        businessImpact: 'informational',
        priority: 'medium'
      }
    };

    // Temporal alignment for news search
    this.temporalMappings = this.config.temporalMappings || {
      'Month-to-Date': {
        newsTimeframe: 'past_month',
        searchPeriod: '30d',
        relevanceWindow: 'current_month'
      },
      'Year-to-Date': {
        newsTimeframe: 'past_year',
        searchPeriod: '365d',
        relevanceWindow: 'current_year'
      },
      'Quarter-to-Date': {
        newsTimeframe: 'past_quarter',
        searchPeriod: '90d',
        relevanceWindow: 'current_quarter'
      }
    };

    // News source credibility mapping
    this.sourceCredibility = this.config.sourceCredibility || {
      'tier1': {
        sources: ['Reuters', 'Bloomberg', 'Financial Times', 'Wall Street Journal', 'Associated Press'],
        weight: 1.0,
        reliability: 'high'
      },
      'tier2': {
        sources: ['CNBC', 'BBC News', 'CNN Business', 'MarketWatch', 'Forbes'],
        weight: 0.8,
        reliability: 'medium-high'
      },
      'tier3': {
        sources: ['Yahoo Finance', 'Business Insider', 'TechCrunch'],
        weight: 0.6,
        reliability: 'medium'
      }
    };
    
    // Processing statistics
    this.stats = {
      totalProcessed: 0,
      successfulQueries: 0,
      errors: 0,
      startTime: null,
      endTime: null
    };
  }

  // Create default logger
  createDefaultLogger() {
    return {
      info: (msg) => console.log(`[INFO] ${msg}`),
      warn: (msg) => console.warn(`[WARN] ${msg}`),
      error: (msg) => console.error(`[ERROR] ${msg}`),
      debug: (msg) => console.log(`[DEBUG] ${msg}`)
    };
  }

  // Load configuration with defaults
  loadConfiguration(config) {
    return {
      batchSize: 1000,
      enableValidation: true,
      logLevel: 'info',
      maxQueriesPerRecord: 10,
      ...config
    };
  }

  // Generate search queries for each dimension of your data
  generateSearchQueries(standardizedRecord) {
    try {
      const queries = [];

    // Location-based queries
    if (standardizedRecord.primary_location || standardizedRecord.L4_clean_name) {
      const location = standardizedRecord.primary_location || standardizedRecord.L4_clean_name;
      const locationVariants = this.entityMappings.locationEntities[location] || [location];
      
      locationVariants.forEach(variant => {
        queries.push({
          type: 'location',
          dimension: 'geography',
          query: `"${variant}" AND (banking OR finance OR markets)`,
          entity: location,
          priority: 'high'
        });
      });
    }

    // Product-based queries
    if (standardizedRecord.product_category || standardizedRecord.L3_clean_name) {
      const product = standardizedRecord.product_category || standardizedRecord.L3_clean_name;
      const productVariants = this.entityMappings.productEntities[product] || [product];
      
      productVariants.forEach(variant => {
        queries.push({
          type: 'product',
          dimension: 'business_line',
          query: `"${variant}" AND (banking OR finance)`,
          entity: product,
          priority: 'high'
        });
      });
    }

    // Account/Financial term queries
    if (standardizedRecord.account_type || standardizedRecord.L3_clean_name) {
      const account = standardizedRecord.account_type || standardizedRecord.L3_clean_name;
      const accountVariants = this.entityMappings.accountEntities[account] || [account];
      
      accountVariants.forEach(variant => {
        queries.push({
          type: 'financial',
          dimension: 'account',
          query: `"${variant}" AND (regulation OR Basel OR IFRS)`,
          entity: account,
          priority: 'medium'
        });
      });
    }

    // Legal entity queries
    if (standardizedRecord.base_entity) {
      queries.push({
        type: 'entity',
        dimension: 'legal',
        query: `"${standardizedRecord.base_entity}" AND (financial OR banking OR entity)`,
        entity: standardizedRecord.base_entity,
        priority: 'medium'
      });
    }

    // Measurement/Reporting queries
    if (standardizedRecord.measure_category) {
      queries.push({
        type: 'measurement',
        dimension: 'reporting',
        query: `"${standardizedRecord.measure_category}" AND (reporting OR measurement OR accounting)`,
        entity: standardizedRecord.measure_category,
        priority: 'low'
      });
    }

    return queries;
    } catch (error) {
      this.logger.error(`Error generating search queries: ${error.message}`);
      return [];
    }
  }

  // Create semantic search terms for modern news APIs
  createSemanticSearchTerms(standardizedRecord) {
    try {
      const semanticTerms = {
      entities: [],
      concepts: [],
      keywords: [],
      locations: [],
      organizations: []
    };

    // Extract entities for semantic search
    if (standardizedRecord.primary_location) {
      semanticTerms.locations.push(standardizedRecord.primary_location);
      semanticTerms.entities.push({
        type: 'LOCATION',
        value: standardizedRecord.primary_location,
        confidence: 0.9
      });
    }

    if (standardizedRecord.base_entity) {
      semanticTerms.organizations.push(standardizedRecord.base_entity);
      semanticTerms.entities.push({
        type: 'ORGANIZATION',
        value: standardizedRecord.base_entity,
        confidence: 0.8
      });
    }

    // Add business concepts
    if (standardizedRecord.product_category) {
      semanticTerms.concepts.push(standardizedRecord.product_category);
      semanticTerms.keywords.push(...(this.entityMappings.productEntities[standardizedRecord.product_category] || []));
    }

    if (standardizedRecord.account_type) {
      semanticTerms.concepts.push(standardizedRecord.account_type);
      semanticTerms.keywords.push(...(this.entityMappings.accountEntities[standardizedRecord.account_type] || []));
    }

    return semanticTerms;
    } catch (error) {
      this.logger.error(`Error creating semantic search terms: ${error.message}`);
      return {
        entities: [],
        concepts: [],
        keywords: [],
        locations: [],
        organizations: []
      };
    }
  }

  // Generate API-specific search parameters
  generateAPISearchParams(record, apiType = 'generic') {
    try {
      switch (apiType.toLowerCase()) {
      case 'perplexity':
        return this.generatePerplexityParams(record);
      case 'worldnews':
        return this.generateWorldNewsParams(record);
      case 'newsapi':
        return this.generateNewsAPIParams(record);
      case 'perigon':
        return this.generatePerigonParams(record);
      default:
        return this.generateGenericParams(record);
    }
    } catch (error) {
      this.logger.error(`Error generating API search params: ${error.message}`);
      return {};
    }
  }

  generatePerplexityParams(record) {
    const queries = this.generateSearchQueries(record);
    return {
      query: queries.map(q => q.query).join(' OR '),
      focus: 'academic', // For more detailed analysis
      search_type: 'web',
      recency: 'month' // Align with MTD if applicable
    };
  }

  generateWorldNewsParams(record) {
    const semanticTerms = this.createSemanticSearchTerms(record);
    return {
      text: semanticTerms.keywords.slice(0, 5).join(' OR '),
      entities: semanticTerms.entities.map(e => `${e.type}:${e.value}`).join(','),
      'source-countries': record.L4_iso2 ? record.L4_iso2.toLowerCase() : null,
      language: 'en',
      'earliest-publish-date': this.getEarliestDate(record),
      'sort-by': 'relevance'
    };
  }

  generateNewsAPIParams(record) {
    const queries = this.generateSearchQueries(record);
    return {
      q: queries.filter(q => q.priority === 'high').map(q => q.query).join(' OR '),
      language: 'en',
      sortBy: 'relevancy',
      from: this.getFromDate(record),
      domains: this.sourceCredibility.tier1.sources.join(',').toLowerCase().replace(/\s+/g, '')
    };
  }

  generatePerigonParams(record) {
    const semanticTerms = this.createSemanticSearchTerms(record);
    return {
      q: semanticTerms.keywords.slice(0, 3).join(' AND '),
      location: semanticTerms.locations[0] || null,
      company: record.base_entity || null,
      from: this.getFromDate(record),
      size: 50,
      'enriched-data': true // For AI analysis
    };
  }

  // Helper methods for date calculations
  getFromDate(record) {
    if (record.version_standardized?.includes('Month-to-Date')) {
      const now = new Date();
      return new Date(now.getFullYear(), now.getMonth(), 1).toISOString().split('T')[0];
    }
    return new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
  }

  getEarliestDate(record) {
    if (record.reporting_year) {
      return `${record.reporting_year}-01-01`;
    }
    return this.getFromDate(record);
  }

  // Analyze news relevance to your business taxonomy
  analyzeNewsRelevance(newsArticle, businessRecord) {
    try {
      const relevanceScore = {
      location: 0,
      product: 0,
      account: 0,
      book: 0,
      measure: 0,
      overall: 0
    };

    // Location relevance
    if (businessRecord.primary_location) {
      const locationVariants = this.entityMappings.locationEntities[businessRecord.primary_location] || [];
      const locationMentions = locationVariants.filter(variant => 
        newsArticle.title?.toLowerCase().includes(variant.toLowerCase()) ||
        newsArticle.content?.toLowerCase().includes(variant.toLowerCase())
      ).length;
      relevanceScore.location = Math.min(locationMentions * 0.3, 1.0);
    }

    // Product relevance
    if (businessRecord.product_category) {
      const productVariants = this.entityMappings.productEntities[businessRecord.product_category] || [];
      const productMentions = productVariants.filter(variant => 
        newsArticle.title?.toLowerCase().includes(variant.toLowerCase()) ||
        newsArticle.content?.toLowerCase().includes(variant.toLowerCase())
      ).length;
      relevanceScore.product = Math.min(productMentions * 0.4, 1.0);
    }

    // Calculate overall relevance
    relevanceScore.overall = (
      relevanceScore.location * 0.3 +
      relevanceScore.product * 0.4 +
      relevanceScore.account * 0.2 +
      relevanceScore.book * 0.05 +
      relevanceScore.measure * 0.05
    );

    return {
      relevanceScore,
      priority: relevanceScore.overall > 0.7 ? 'high' : 
                relevanceScore.overall > 0.4 ? 'medium' : 'low',
      matchedEntities: this.extractMatchedEntities(newsArticle, businessRecord)
    };
    } catch (error) {
      this.logger.error(`Error analyzing news relevance: ${error.message}`);
      return {
        relevanceScore: { overall: 0 },
        priority: 'low',
        matchedEntities: []
      };
    }
  }

  extractMatchedEntities(newsArticle, businessRecord) {
    try {
      const matched = [];
    
    // Check all entity mappings for matches
    Object.entries(this.entityMappings).forEach(([category, entities]) => {
      Object.entries(entities).forEach(([standardized, variants]) => {
        variants.forEach(variant => {
          if (newsArticle.title?.toLowerCase().includes(variant.toLowerCase()) ||
              newsArticle.content?.toLowerCase().includes(variant.toLowerCase())) {
            matched.push({
              category,
              standardizedTerm: standardized,
              matchedVariant: variant,
              businessDimension: this.mapCategoryToDimension(category)
            });
          }
        });
      });
    });

    return matched;
    } catch (error) {
      this.logger.error(`Error extracting matched entities: ${error.message}`);
      return [];
    }
  }

  mapCategoryToDimension(category) {
    const mapping = {
      'locationEntities': 'location',
      'productEntities': 'product', 
      'accountEntities': 'account',
      'industryEntities': 'product'
    };
    return mapping[category] || 'unknown';
  }

  // Generate comprehensive news search configuration
  generateNewsSearchConfig(standardizedDatasets) {
    try {
      const config = {
      entityMappings: this.entityMappings,
      searchStrategies: {},
      apiConfigurations: {},
      relevanceThresholds: {
        high: 0.7,
        medium: 0.4,
        low: 0.1
      },
      updateFrequency: {
        realtime: ['high_priority_locations', 'major_products'],
        hourly: ['medium_priority_entities'],
        daily: ['low_priority_entities']
      }
    };

    // Generate search strategies for each dataset
    ['locations', 'accounts', 'products', 'books', 'measures'].forEach(dataset => {
      config.searchStrategies[dataset] = this.generateDatasetSearchStrategy(dataset);
    });

    return config;
    } catch (error) {
      this.logger.error(`Error generating news search config: ${error.message}`);
      return {
        entityMappings: this.entityMappings,
        searchStrategies: {},
        apiConfigurations: {},
        relevanceThresholds: {
          high: 0.7,
          medium: 0.4,
          low: 0.1
        }
      };
    }
  }

  generateDatasetSearchStrategy(dataset) {
    const strategies = {
      locations: {
        primaryQueries: ['economic indicators', 'market conditions', 'regulatory changes'],
        entityTypes: ['LOCATION', 'GPE'],
        updateFrequency: 'hourly',
        relevanceWeight: 0.3
      },
      products: {
        primaryQueries: ['industry trends', 'market developments', 'regulatory updates'],
        entityTypes: ['PRODUCT', 'SERVICE', 'INDUSTRY'],
        updateFrequency: 'realtime',
        relevanceWeight: 0.4
      },
      accounts: {
        primaryQueries: ['financial regulations', 'accounting standards', 'risk management'],
        entityTypes: ['FINANCIAL_INSTRUMENT', 'REGULATION'],
        updateFrequency: 'daily',
        relevanceWeight: 0.2
      },
      books: {
        primaryQueries: ['corporate structure', 'legal entities', 'consolidation'],
        entityTypes: ['ORGANIZATION', 'LEGAL_ENTITY'],
        updateFrequency: 'daily',
        relevanceWeight: 0.05
      },
      measures: {
        primaryQueries: ['reporting standards', 'measurement frameworks', 'accounting methods'],
        entityTypes: ['STANDARD', 'METHODOLOGY'],
        updateFrequency: 'weekly',
        relevanceWeight: 0.05
      }
    };

    return strategies[dataset] || strategies.products;
  }

  // Process a batch of records
  async processBatch(batch, startIndex) {
    const results = [];
    
    for (let i = 0; i < batch.length; i++) {
      const record = batch[i];
      
      try {
        const searchQueries = this.generateSearchQueries(record);
        const semanticTerms = this.createSemanticSearchTerms(record);
        
        const result = {
          original_record: record,
          search_queries: searchQueries.slice(0, this.config.maxQueriesPerRecord),
          semantic_terms: semanticTerms,
          api_params: {
            perplexity: this.generateAPISearchParams(record, 'perplexity'),
            worldnews: this.generateAPISearchParams(record, 'worldnews'),
            newsapi: this.generateAPISearchParams(record, 'newsapi'),
            perigon: this.generateAPISearchParams(record, 'perigon')
          }
        };
        
        results.push(result);
        this.stats.successfulQueries++;
      } catch (error) {
        this.logger.error(`Error processing record: ${error.message}`);
        this.stats.errors++;
        
        // Add error record
        results.push({
          original_record: record,
          error: error.message
        });
      }
    }
    
    return results;
  }

  // Process entire dataset with batch processing
  async standardizeDataset(data) {
    this.logger.info(`Starting news search standardization for ${data.length} records`);
    this.stats.startTime = Date.now();
    this.stats.totalProcessed = data.length;
    
    const results = [];
    const batchSize = this.config.batchSize;
    
    // Process in batches
    for (let i = 0; i < data.length; i += batchSize) {
      const batch = data.slice(i, Math.min(i + batchSize, data.length));
      const batchNumber = Math.floor(i / batchSize) + 1;
      const totalBatches = Math.ceil(data.length / batchSize);
      
      this.logger.info(`Processing batch ${batchNumber}/${totalBatches} (${batch.length} records)`);
      
      try {
        const batchResults = await this.processBatch(batch, i);
        results.push(...batchResults);
      } catch (error) {
        this.logger.error(`Error processing batch ${batchNumber}: ${error.message}`);
        this.stats.errors += batch.length;
      }
    }
    
    this.stats.endTime = Date.now();
    const duration = (this.stats.endTime - this.stats.startTime) / 1000;
    
    this.logger.info(`News search standardization completed in ${duration.toFixed(2)} seconds`);
    this.logger.info(`Successfully processed: ${this.stats.successfulQueries}/${this.stats.totalProcessed}`);
    if (this.stats.errors > 0) {
      this.logger.warn(`Errors encountered: ${this.stats.errors}`);
    }
    
    return results;
  }

  // Generate standardization report
  generateReport(originalData, standardizedData) {
    try {
      const report = {
        totalRecords: originalData.length,
        processingStats: {
          successfulQueries: this.stats.successfulQueries,
          errors: this.stats.errors,
          processingTime: this.stats.endTime ? ((this.stats.endTime - this.stats.startTime) / 1000).toFixed(2) + ' seconds' : 'N/A'
        },
        queryStats: {
          totalQueries: standardizedData.reduce((sum, r) => sum + (r.search_queries?.length || 0), 0),
          avgQueriesPerRecord: (standardizedData.reduce((sum, r) => sum + (r.search_queries?.length || 0), 0) / standardizedData.length).toFixed(2),
          locationQueries: standardizedData.reduce((sum, r) => sum + (r.search_queries?.filter(q => q.type === 'location').length || 0), 0),
          productQueries: standardizedData.reduce((sum, r) => sum + (r.search_queries?.filter(q => q.type === 'product').length || 0), 0),
          financialQueries: standardizedData.reduce((sum, r) => sum + (r.search_queries?.filter(q => q.type === 'financial').length || 0), 0)
        },
        apiCoverage: {
          perplexity: standardizedData.filter(r => r.api_params?.perplexity).length,
          worldnews: standardizedData.filter(r => r.api_params?.worldnews).length,
          newsapi: standardizedData.filter(r => r.api_params?.newsapi).length,
          perigon: standardizedData.filter(r => r.api_params?.perigon).length
        },
        recommendations: []
      };

      // Generate recommendations
      if (report.queryStats.avgQueriesPerRecord < 2) {
        report.recommendations.push('Consider enriching source data to generate more targeted queries');
      }
      
      if (report.processingStats.errors > 0) {
        report.recommendations.push(`${report.processingStats.errors} records encountered errors during processing`);
      }

      this.logger.info('Generated standardization report');
      return report;
    } catch (error) {
      this.logger.error(`Error generating report: ${error.message}`);
      return {
        error: error.message,
        totalRecords: originalData ? originalData.length : 0
      };
    }
  }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = NewsSearchStandardizer;
} else {
  window.NewsSearchStandardizer = NewsSearchStandardizer;
}