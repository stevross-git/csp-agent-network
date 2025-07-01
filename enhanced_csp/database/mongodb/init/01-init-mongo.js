// MongoDB Initialization Script

// Switch to csp_nosql database
db = db.getSiblingDB('csp_nosql');

// Create collections with validation
db.createCollection('events', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['event_type', 'timestamp', 'data'],
            properties: {
                event_type: {
                    bsonType: 'string',
                    description: 'Type of event'
                },
                timestamp: {
                    bsonType: 'date',
                    description: 'Event timestamp'
                },
                data: {
                    bsonType: 'object',
                    description: 'Event data'
                }
            }
        }
    }
});

db.createCollection('logs', {
    capped: true,
    size: 104857600, // 100MB
    max: 1000000
});

db.createCollection('configurations', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['name', 'version', 'config'],
            properties: {
                name: {
                    bsonType: 'string',
                    description: 'Configuration name'
                },
                version: {
                    bsonType: 'string',
                    description: 'Configuration version'
                },
                config: {
                    bsonType: 'object',
                    description: 'Configuration data'
                }
            }
        }
    }
});

// Create indexes
db.events.createIndex({ timestamp: -1 });
db.events.createIndex({ event_type: 1, timestamp: -1 });
db.logs.createIndex({ timestamp: -1 });
db.configurations.createIndex({ name: 1, version: 1 }, { unique: true });

// Create user for application
db.createUser({
    user: 'csp_app',
    pwd: 'csp_app_pass_2024!',
    roles: [
        { role: 'readWrite', db: 'csp_nosql' }
    ]
});
