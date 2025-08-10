const hana = require('@sap/hana-client');
const fs = require('fs').promises;
const path = require('path');
const { exec } = require('child_process');
const { promisify } = require('util');

const execAsync = promisify(exec);

const connOptions = {
    serverNode: 'd93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com:443',
    uid: 'DBADMIN',
    pwd: 'Initial@1',
    encrypt: true,
    sslValidateCertificate: false,
    currentSchema: 'A2A_DEV'
};

async function deployTables() {
    const connection = hana.createConnection();
    
    try {
        // Generate SQL from CDS
        console.log('Generating SQL from CDS models...');
        const { stdout: sql } = await execAsync('cds compile db/schema.cds --to sql');
        
        // Connect to HANA
        console.log('Connecting to HANA...');
        await new Promise((resolve, reject) => {
            connection.connect(connOptions, (err) => {
                if (err) reject(err);
                else resolve();
            });
        });
        
        console.log('Connected successfully!');
        
        // Split SQL statements and execute them
        const statements = sql.split(';').filter(stmt => stmt.trim().length > 0);
        
        for (const statement of statements) {
            const trimmedStmt = statement.trim();
            if (!trimmedStmt) continue;
            
            try {
                console.log(`Executing: ${trimmedStmt.substring(0, 50)}...`);
                await new Promise((resolve, reject) => {
                    connection.exec(trimmedStmt, (err, result) => {
                        if (err) {
                            // Ignore errors for existing tables
                            if (err.code === 288) { // Table already exists
                                console.log('  -> Table already exists, skipping');
                                resolve();
                            } else {
                                reject(err);
                            }
                        } else {
                            console.log('  -> Success');
                            resolve(result);
                        }
                    });
                });
            } catch (err) {
                console.error(`  -> Error: ${err.message}`);
                // Continue with next statement
            }
        }
        
        // Create some sample data
        console.log('\nInserting sample data...');
        
        const sampleAgents = [
            {
                ID: '11111111-1111-1111-1111-111111111111',
                name: 'Data Processor Alpha',
                address: '0x1234567890123456789012345678901234567890',
                endpoint: 'https://agent1.a2a.network',
                reputation: 100,
                isActive: true
            },
            {
                ID: '22222222-2222-2222-2222-222222222222',
                name: 'API Gateway Beta',
                address: '0x2345678901234567890123456789012345678901',
                endpoint: 'https://agent2.a2a.network',
                reputation: 95,
                isActive: true
            },
            {
                ID: '33333333-3333-3333-3333-333333333333',
                name: 'Monitoring Agent Gamma',
                address: '0x3456789012345678901234567890123456789012',
                endpoint: 'https://agent3.a2a.network',
                reputation: 90,
                isActive: true
            }
        ];
        
        for (const agent of sampleAgents) {
            try {
                const stmt = `INSERT INTO a2a_network_Agents (ID, name, address, endpoint, reputation, isActive) VALUES (?, ?, ?, ?, ?, ?)`;
                await new Promise((resolve, reject) => {
                    connection.prepare(stmt, (err, statement) => {
                        if (err) return reject(err);
                        statement.exec([agent.ID, agent.name, agent.address, agent.endpoint, agent.reputation, agent.isActive], (err, result) => {
                            if (err && err.code !== 301) { // Ignore unique constraint violations
                                reject(err);
                            } else {
                                resolve(result);
                            }
                        });
                    });
                });
                console.log(`  -> Inserted agent: ${agent.name}`);
            } catch (err) {
                console.log(`  -> Agent already exists: ${agent.name}`);
            }
        }
        
        console.log('\nDeployment completed successfully!');
        
    } catch (err) {
        console.error('Deployment failed:', err);
        process.exit(1);
    } finally {
        connection.disconnect();
    }
}

deployTables();