/**
 * Script to seed test services into the database
 */

const cds = require('@sap/cds');

async function seedTestServices() {
    try {
        log.debug('🚀 Seeding test services...');
        
        // Load model and connect to database
        await cds.load('*');
        const db = await cds.connect.to('db');
        
        const serviceTableName = 'a2a_network_Services';
        
        // Get some agent IDs to use as providers
        const agentIds = await db.run(`SELECT ID FROM a2a_network_Agents LIMIT 8`);
        
        // Create 8 test services
        const services = [];
        const serviceTypes = [
            'Data Analysis', 'Machine Learning', 'Natural Language Processing',
            'Image Recognition', 'Predictive Analytics', 'Financial Modeling',
            'Risk Assessment', 'Market Research'
        ];
        
        for (let i = 1; i <= 8; i++) {
            services.push({
                ID: cds.utils.uuid(),
                provider_ID: agentIds[i-1]?.ID || agentIds[0]?.ID, // Use existing agent as provider
                name: serviceTypes[i-1],
                description: `Advanced ${serviceTypes[i-1]} service powered by AI`,
                pricePerCall: Math.floor(Math.random() * 100) + 10, // 10-110
                isActive: 1,
                category: 'AI_SERVICES',
                createdAt: new Date().toISOString(),
                modifiedAt: new Date().toISOString()
            });
        }
        
        // Clear existing services first
        await db.run(`DELETE FROM ${serviceTableName}`);
        log.debug('🧹 Cleared existing services');
        
        // Insert new services using raw SQL (without currency_code to avoid FK constraint)
        const insertSQL = `INSERT INTO ${serviceTableName} (ID, provider_ID, name, description, pricePerCall, isActive, category, createdAt, modifiedAt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`;
        
        for (const service of services) {
            await db.run(insertSQL, [
                service.ID,
                service.provider_ID,
                service.name,
                service.description,
                service.pricePerCall,
                service.isActive,
                service.category,
                service.createdAt,
                service.modifiedAt
            ]);
        }
        
        log.info(`✅ Successfully seeded ${services.length} test services`);
        
        // Verify the count
        const countResult = await db.run(`SELECT COUNT(*) as total FROM ${serviceTableName}`);
        log.debug(`📊 Total services in database: ${countResult[0]?.total || 0}`);
        
        process.exit(0);
        
    } catch (error) {
        console.error('❌ Error seeding test services:', error);
        process.exit(1);
    }
}

// Run the seeding
seedTestServices();