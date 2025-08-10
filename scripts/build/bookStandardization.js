// Financial Book Classification Standardization Framework
// Phase 1: Legal Entity & Adjustment Category Standardization

class BookStandardizer {
  constructor(config = {}) {
    // Initialize logger
    this.logger = config.logger || this.createDefaultLogger();
    
    // Load configuration with defaults
    this.config = this.loadConfiguration(config);
    
    // Financial book and adjustment abbreviation mappings
    this.abbreviationMappings = this.config.abbreviationMappings || {
      // Book/Entity abbreviations
      'NEWF': 'New Financial',
      'GRP': 'Group',
      'ADJ': 'Adjustment',
      'Adj': 'Adjustment',
      'MAN': 'Manual',
      'AUTO': 'Automatic',
      'SYS': 'System',
      
      // Legal Entity abbreviations
      'LTD': 'Limited',
      'LLC': 'Limited Liability Company',
      'PLC': 'Public Limited Company',
      'INC': 'Incorporated',
      'CORP': 'Corporation',
      'CO': 'Company',
      
      // Adjustment type abbreviations
      'CONSOL': 'Consolidation',
      'ELIM': 'Elimination',
      'RECLASS': 'Reclassification',
      'ACCR': 'Accrual',
      'DEFER': 'Deferral',
      'FX': 'Foreign Exchange',
      'MTM': 'Mark to Market',
      'PVA': 'Prudent Valuation Adjustment',
      'CVA': 'Credit Valuation Adjustment',
      'DVA': 'Debt Valuation Adjustment',
      'FVA': 'Funding Valuation Adjustment',
      
      // Regulatory abbreviations
      'IFRS': 'International Financial Reporting Standards',
      'GAAP': 'Generally Accepted Accounting Principles',
      'BASEL': 'Basel Regulatory Framework',
      'CRD': 'Capital Requirements Directive',
      'CRR': 'Capital Requirements Regulation'
    };

    // Book classification types
    this.bookClassifications = this.config.bookClassifications || {
      // Legal Entity Types
      'Legal Entity': {
        type: 'Legal Entity',
        subtype: 'Primary Entity',
        purpose: 'Legal Structure',
        regulatory_scope: 'Full Reporting',
        consolidation_method: 'Full Consolidation'
      },
      'Subsidiary': {
        type: 'Legal Entity',
        subtype: 'Subsidiary Entity',
        purpose: 'Legal Structure',
        regulatory_scope: 'Full Reporting',
        consolidation_method: 'Full Consolidation'
      },
      'Branch': {
        type: 'Legal Entity',
        subtype: 'Branch Office',
        purpose: 'Geographic Presence',
        regulatory_scope: 'Local Reporting',
        consolidation_method: 'Full Consolidation'
      },
      
      // Adjustment Categories
      'Group Adjustment': {
        type: 'Consolidation Adjustment',
        subtype: 'Group Level Adjustment',
        purpose: 'Consolidation Process',
        regulatory_scope: 'Group Reporting',
        consolidation_method: 'Adjustment Entry'
      },
      'Manual Adjustment': {
        type: 'Operational Adjustment',
        subtype: 'Manual Entry',
        purpose: 'Data Correction',
        regulatory_scope: 'Internal Control',
        consolidation_method: 'Manual Override'
      },
      'Automatic Adjustment': {
        type: 'Operational Adjustment',
        subtype: 'System Generated',
        purpose: 'Automated Process',
        regulatory_scope: 'System Control',
        consolidation_method: 'Automated Process'
      },
      'Elimination Entry': {
        type: 'Consolidation Adjustment',
        subtype: 'Intercompany Elimination',
        purpose: 'Remove Intercompany',
        regulatory_scope: 'Group Reporting',
        consolidation_method: 'Elimination'
      },
      'Reclassification': {
        type: 'Presentation Adjustment',
        subtype: 'Account Reclassification',
        purpose: 'Presentation Change',
        regulatory_scope: 'Reporting Format',
        consolidation_method: 'Reclassification'
      }
    };

    // Regulatory framework mappings
    this.regulatoryFrameworks = this.config.regulatoryFrameworks || {
      'IFRS': {
        framework: 'International Financial Reporting Standards',
        scope: 'Global',
        authority: 'IASB',
        application: 'Financial Reporting'
      },
      'Basel': {
        framework: 'Basel Regulatory Framework',
        scope: 'Banking',
        authority: 'Basel Committee',
        application: 'Risk Management'
      },
      'Local GAAP': {
        framework: 'Local Generally Accepted Accounting Principles',
        scope: 'Jurisdictional',
        authority: 'Local Regulator',
        application: 'Local Reporting'
      }
    };

    // Book hierarchy validation rules
    this.hierarchyRules = this.config.hierarchyRules || {
      'Entity Books': ['Legal Entity', 'Subsidiary', 'Branch'],
      'Adjustment Books': ['Group Adjustment', 'Manual Adjustment', 'Elimination Entry'],
      'Process Books': ['Consolidation', 'Reporting', 'Regulatory']
    };
    
    // Processing statistics
    this.stats = {
      totalProcessed: 0,
      successfulStandardizations: 0,
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
      ...config
    };
  }

  // Clean and standardize book names
  cleanBookName(bookName) {
    try {
      if (!bookName) return '';
      
      let cleaned = bookName
        .replace(/\s+/g, ' ')  // Remove extra spaces
        .trim();

    // Expand abbreviations
    Object.entries(this.abbreviationMappings).forEach(([abbr, expansion]) => {
      // Handle various abbreviation formats
      const patterns = [
        new RegExp(`\\b${abbr}\\b`, 'gi'),           // Standalone abbreviation
        new RegExp(`^${abbr}\\s`, 'gi'),             // Start of string
        new RegExp(`\\s${abbr}\\s`, 'gi'),           // Middle of string
        new RegExp(`\\s${abbr}$`, 'gi'),             // End of string
        new RegExp(`${abbr}-`, 'gi'),                // Hyphenated
        new RegExp(`${abbr}_`, 'gi'),                // Underscored
      ];
      
      patterns.forEach(pattern => {
        cleaned = cleaned.replace(pattern, (match) => {
          const prefix = match.startsWith(' ') ? ' ' : '';
          const suffix = match.endsWith(' ') || match.endsWith('-') || match.endsWith('_') ? 
                        match.slice(-1) : '';
          return `${prefix}${expansion}${suffix}`;
        });
      });
    });

    // Standardize common patterns
    cleaned = cleaned
      .replace(/\s*-\s*/g, ' - ')             // Standardize hyphens
      .replace(/\s*_\s*/g, ' ')               // Remove underscores
      .replace(/\s*&\s*/g, ' and ')           // & → and
      .replace(/\(\s*/g, '(')                 // Remove space after (
      .replace(/\s*\)/g, ')')                 // Remove space before )
      .replace(/\badj\b/gi, 'Adjustment')     // Standardize "adj"
      .replace(/\bman\b/gi, 'Manual')         // Standardize "man" 
      .replace(/\bgrp\b/gi, 'Group')          // Standardize "grp"
      .replace(/\bauto\b/gi, 'Automatic')     // Standardize "auto"
      .replace(/\bsys\b/gi, 'System');        // Standardize "sys"

    // Title case with exceptions
    cleaned = cleaned.replace(/\b\w+/g, (word) => {
      // Keep certain words lowercase
      const lowercaseWords = ['and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'the', 'a', 'an'];
      // Keep certain abbreviations uppercase
      const uppercaseWords = ['IFRS', 'GAAP', 'USD', 'EUR', 'GBP', 'FX', 'ID', 'PLC', 'LLC'];
      
      if (lowercaseWords.includes(word.toLowerCase())) {
        return word.toLowerCase();
      } else if (uppercaseWords.includes(word.toUpperCase())) {
        return word.toUpperCase();
      } else {
        return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
      }
    });

    // Ensure first word is always capitalized
    cleaned = cleaned.charAt(0).toUpperCase() + cleaned.slice(1);

    return cleaned;
    } catch (error) {
      this.logger.error(`Error cleaning book name: ${error.message}`);
      return bookName || '';
    }
  }

  // Extract entity information
  extractEntityInfo(bookName) {
    try {
      const cleanName = this.cleanBookName(bookName);
      
      // Identify base entity
      let baseEntity = null;
      let entityType = 'Unknown';
    
    // Common entity patterns
    if (cleanName.includes('New Financial')) {
      baseEntity = 'New Financial';
      entityType = 'Legal Entity';
    }
    
    // Extract adjustment type
    let adjustmentType = null;
    if (cleanName.toLowerCase().includes('group') && cleanName.toLowerCase().includes('adjustment')) {
      adjustmentType = 'Group Adjustment';
    } else if (cleanName.toLowerCase().includes('manual') && cleanName.toLowerCase().includes('adjustment')) {
      adjustmentType = 'Manual Adjustment';
    } else if (cleanName.toLowerCase().includes('elimination')) {
      adjustmentType = 'Elimination Entry';
    } else if (cleanName.toLowerCase().includes('reclassification')) {
      adjustmentType = 'Reclassification';
    }
    
    return {
      baseEntity,
      entityType,
      adjustmentType,
      isAdjustmentBook: !!adjustmentType,
      isMainEntity: !adjustmentType
    };
    } catch (error) {
      this.logger.error(`Error extracting entity info: ${error.message}`);
      return {
        baseEntity: null,
        entityType: 'Unknown',
        adjustmentType: null,
        isAdjustmentBook: false,
        isMainEntity: false
      };
    }
  }

  // Classify book type and characteristics
  classifyBook(cleanBookName, entityInfo) {
    try {
      // Try to match against known classifications
      for (const [pattern, classification] of Object.entries(this.bookClassifications)) {
      if (cleanBookName.toLowerCase().includes(pattern.toLowerCase()) ||
          entityInfo.adjustmentType === pattern) {
        return classification;
      }
    }

    // Fallback classification based on entity info
    if (entityInfo.isMainEntity) {
      return {
        type: 'Legal Entity',
        subtype: 'Primary Entity',
        purpose: 'Legal Structure',
        regulatory_scope: 'Full Reporting',
        consolidation_method: 'Full Consolidation'
      };
    } else if (entityInfo.isAdjustmentBook) {
      return {
        type: 'Consolidation Adjustment',
        subtype: 'Adjustment Entry',
        purpose: 'Consolidation Process',
        regulatory_scope: 'Group Reporting',
        consolidation_method: 'Adjustment Entry'
      };
    }

    // Default classification
    return {
      type: 'Unknown',
      subtype: 'Unknown',
      purpose: 'Unknown',
      regulatory_scope: 'Unknown',
      consolidation_method: 'Unknown'
    };
    } catch (error) {
      this.logger.error(`Error classifying book: ${error.message}`);
      return {
        type: 'Unknown',
        subtype: 'Unknown',
        purpose: 'Unknown',
        regulatory_scope: 'Unknown',
        consolidation_method: 'Unknown'
      };
    }
  }

  // Generate book codes
  generateBookCode(cleanBookName, index) {
    try {
      const entityInfo = this.extractEntityInfo(cleanBookName);
      
      let typeCode = 'UN'; // Unknown
      if (entityInfo.isMainEntity) {
        typeCode = 'ME'; // Main Entity
      } else if (entityInfo.adjustmentType === 'Group Adjustment') {
        typeCode = 'GA'; // Group Adjustment
      } else if (entityInfo.adjustmentType === 'Manual Adjustment') {
        typeCode = 'MA'; // Manual Adjustment
      }
      
      const sequenceNumber = String(index + 1).padStart(2, '0');
      return `${typeCode}${sequenceNumber}`;
    } catch (error) {
      this.logger.error(`Error generating book code: ${error.message}`);
      return `UN${String(index + 1).padStart(2, '0')}`;
    }
  }

  // Validate book consistency
  validateBook(bookName) {
    try {
      const issues = [];
      
      if (!this.config.enableValidation) {
        return issues;
      }
      
      // Check for incomplete abbreviations
      const commonAbbreviations = ['GRP', 'ADJ', 'MAN', 'AUTO'];
      commonAbbreviations.forEach(abbr => {
        if (bookName.includes(abbr) && !this.abbreviationMappings[abbr]) {
          issues.push(`Unrecognized abbreviation: ${abbr}`);
        }
      });
      
      // Check for naming consistency
      if (bookName.includes('-') && !bookName.includes(' - ')) {
        issues.push('Inconsistent hyphen spacing');
      }
      
      return issues;
    } catch (error) {
      this.logger.error(`Error validating book: ${error.message}`);
      return [];
    }
  }

  // Process a batch of records
  async processBatch(batch, startIndex) {
    const results = [];
    
    for (let i = 0; i < batch.length; i++) {
      const row = batch[i];
      const index = startIndex + i;
      
      try {
        const result = {
          // Original data preserved
          original_row_number: row._row_number,
          original_book_name: row.Books
        };

        // Clean and standardize book name
        const cleanName = this.cleanBookName(row.Books);
        const entityInfo = this.extractEntityInfo(row.Books);
        const classification = this.classifyBook(cleanName, entityInfo);

        // Add standardized fields
        result.clean_book_name = cleanName;
        result.base_entity = entityInfo.baseEntity;
        result.entity_type = entityInfo.entityType;
        result.adjustment_type = entityInfo.adjustmentType;
        result.is_adjustment_book = entityInfo.isAdjustmentBook;
        result.is_main_entity = entityInfo.isMainEntity;

        // Classification details
        result.book_type = classification.type;
        result.book_subtype = classification.subtype;
        result.book_purpose = classification.purpose;
        result.regulatory_scope = classification.regulatory_scope;
        result.consolidation_method = classification.consolidation_method;

        // Generate book code
        result.generated_book_code = this.generateBookCode(cleanName, index);

        // Validation
        const validationIssues = this.validateBook(row.Books);
        result.validation_issues = validationIssues.join('; ') || null;

        // Quality assessment
        result.standardization_quality = this.assessStandardizationQuality(result);

        results.push(result);
        this.stats.successfulStandardizations++;
      } catch (error) {
        this.logger.error(`Error processing row ${row._row_number}: ${error.message}`);
        this.stats.errors++;
        
        // Add error record
        results.push({
          original_row_number: row._row_number,
          original_book_name: row.Books,
          error: error.message,
          standardization_quality: 'Error'
        });
      }
    }
    
    return results;
  }

  // Process entire dataset with batch processing
  async standardizeDataset(data) {
    this.logger.info(`Starting book standardization for ${data.length} records`);
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
    
    this.logger.info(`Book standardization completed in ${duration.toFixed(2)} seconds`);
    this.logger.info(`Successfully processed: ${this.stats.successfulStandardizations}/${this.stats.totalProcessed}`);
    if (this.stats.errors > 0) {
      this.logger.warn(`Errors encountered: ${this.stats.errors}`);
    }
    
    return results;
  }

  // Assess standardization quality
  assessStandardizationQuality(record) {
    try {
      let score = 0;
      let maxScore = 4;

      // Clean naming improvement (1 point)
      if (record.clean_book_name !== record.original_book_name) score += 1;

      // Entity identification (1 point)
      if (record.base_entity) score += 1;

      // Book classification (1 point)
      if (record.book_type !== 'Unknown') score += 1;

      // Validation passed (1 point)
      if (!record.validation_issues) score += 1;

      const percentage = (score / maxScore) * 100;
      
      if (percentage >= 75) return 'Excellent';
      if (percentage >= 50) return 'Good';
      if (percentage >= 25) return 'Fair';
      return 'Needs Review';
    } catch (error) {
      this.logger.error(`Error assessing quality: ${error.message}`);
      return 'Unknown';
    }
  }

  // Generate standardization report
  generateReport(originalData, standardizedData) {
    try {
      const report = {
        totalRecords: originalData.length,
        processingStats: {
          successfulStandardizations: this.stats.successfulStandardizations,
          errors: this.stats.errors,
          processingTime: this.stats.endTime ? ((this.stats.endTime - this.stats.startTime) / 1000).toFixed(2) + ' seconds' : 'N/A'
        },
        standardizationStats: {
          booksWithEntityIdentified: standardizedData.filter(r => r.base_entity).length,
          adjustmentBooks: standardizedData.filter(r => r.is_adjustment_book).length,
          mainEntityBooks: standardizedData.filter(r => r.is_main_entity).length,
          booksClassified: standardizedData.filter(r => r.book_type !== 'Unknown').length
        },
        qualityBreakdown: {
          excellent: standardizedData.filter(r => r.standardization_quality === 'Excellent').length,
          good: standardizedData.filter(r => r.standardization_quality === 'Good').length,
          fair: standardizedData.filter(r => r.standardization_quality === 'Fair').length,
          needsReview: standardizedData.filter(r => r.standardization_quality === 'Needs Review').length,
          errors: standardizedData.filter(r => r.standardization_quality === 'Error').length
        },
        bookTypeBreakdown: {},
        recommendations: []
      };

      // Book type breakdown
      standardizedData.forEach(record => {
        const type = record.book_type || 'Unknown';
        report.bookTypeBreakdown[type] = (report.bookTypeBreakdown[type] || 0) + 1;
      });

      // Generate recommendations
      if (report.standardizationStats.adjustmentBooks > 0) {
        report.recommendations.push('Consider implementing automated adjustment tracking for better auditability');
      }
      
      if (report.qualityBreakdown.needsReview > 0) {
        report.recommendations.push(`${report.qualityBreakdown.needsReview} books need manual review for complete standardization`);
      }

      if (report.qualityBreakdown.errors > 0) {
        report.recommendations.push(`${report.qualityBreakdown.errors} books encountered errors during processing`);
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
  module.exports = BookStandardizer;
} else {
  window.BookStandardizer = BookStandardizer;
}