const hana = require('@sap/hana-client');

const connOptions = {
    serverNode: 'd93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com:443',
    uid: 'DBADMIN',
    pwd: 'Initial@1',
    encrypt: true,
    sslValidateCertificate: process.env.HANA_SSL_VALIDATE_CERTIFICATE !== 'false'
};

const connection = hana.createConnection();

console.log('Connecting to HANA...');

connection.connect(connOptions, (err) => {
    if (err) {
        console.error('Connection failed:', err);
        process.exit(1);
    }

    console.log('Successfully connected to HANA!');

    // Create schema
    console.log('Creating schema A2A_DEV...');
    connection.exec('CREATE SCHEMA A2A_DEV', (err, result) => {
        if (err) {
            if (err.code === 386) {
                console.log('Schema A2A_DEV already exists');
            } else {
                console.error('Failed to create schema:', err);
                connection.disconnect();
                process.exit(1);
            }
        } else {
            console.log('Schema A2A_DEV created successfully');
        }

        // Grant privileges
        console.log('Granting privileges...');
        connection.exec('GRANT ALL PRIVILEGES ON SCHEMA A2A_DEV TO DBADMIN WITH GRANT OPTION', (err, result) => {
            if (err) {
                console.error('Failed to grant privileges:', err);
            } else {
                console.log('Privileges granted successfully');
            }

            connection.disconnect();
            console.log('Setup complete!');
            process.exit(0);
        });
    });
});