const hana = require('@sap/hana-client');

async function testDatabase() {
    const connection = hana.createConnection();
    
    const connOptions = {
        serverNode: 'd93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com:443',
        uid: 'DBADMIN',
        pwd: 'Initial@1',
        encrypt: true,
        sslValidateCertificate: false,
        currentSchema: 'A2A_DEV'
    };
    
    try {
        console.log('Testing HANA Cloud connection...');
        
        await new Promise((resolve, reject) => {
            connection.connect(connOptions, (err) => {
                if (err) reject(err);
                else resolve();
            });
        });
        
        console.log('✅ Connected successfully!');
        
        // Test basic query
        const result = await new Promise((resolve, reject) => {
            connection.exec('SELECT CURRENT_TIMESTAMP FROM DUMMY', (err, result) => {
                if (err) reject(err);
                else resolve(result);
            });
        });
        
        console.log('✅ Basic query successful:', result[0]);
        
        // Check if our table exists
        const tables = await new Promise((resolve, reject) => {
            connection.exec('SELECT TABLE_NAME FROM TABLES WHERE SCHEMA_NAME = \'A2A_DEV\' AND TABLE_NAME LIKE \'A2A_NETWORK%\'', (err, result) => {
                if (err) reject(err);
                else resolve(result);
            });
        });
        
        console.log('✅ Found tables:', tables.map(t => t.TABLE_NAME));
        
        // Test data query
        if (tables.some(t => t.TABLE_NAME === 'A2A_NETWORK_AGENTS')) {
            const agents = await new Promise((resolve, reject) => {
                connection.exec('SELECT ID, NAME FROM A2A_NETWORK_AGENTS LIMIT 3', (err, result) => {
                    if (err) reject(err);
                    else resolve(result);
                });
            });
            
            console.log('✅ Sample agents:', agents);
        }
        
        console.log('✅ All tests passed - HANA is ready!');
        
    } catch (error) {
        console.error('❌ Database test failed:', error.message);
        process.exit(1);
    } finally {
        connection.disconnect();
    }
}

testDatabase();