#!/bin/bash
# Fix indentation issues in backend/main.py

echo "🔧 Fixing indentation in backend/main.py..."

# First, let's restore from backup and apply the changes properly
if [ -f backend/main.py.backup.network.* ]; then
    echo "📝 Restoring from backup..."
    cp backend/main.py.backup.network.* backend/main.py
fi

# Create a properly formatted patch
echo "📝 Creating proper integration patch..."
cat > backend/main_network_integration.py << 'EOF'
# Add this to the imports section after WebSocket dependencies:

# Network Integration
try:
    from backend.network_integration import (
        initialize_network_service,
        shutdown_network_service,
        network_service,
        integrate_websocket_with_network
    )
    NETWORK_AVAILABLE = True
except ImportError:
    logger.warning("Network integration not available")
    NETWORK_AVAILABLE = False
    network_service = None

# Add this in the lifespan function after component registry initialization:
        
        # Initialize network service
        if NETWORK_AVAILABLE:
            await initialize_network_service(app)
            
            # Integrate WebSocket with network
            if WEBSOCKET_AVAILABLE and network_service and network_service.is_initialized:
                integrate_websocket_with_network(connection_manager, network_service)

# Add this in the shutdown section after the shutdown log message:
        
        # Shutdown network service
        if NETWORK_AVAILABLE and network_service:
            await shutdown_network_service()

# Add this to the features dict in the root endpoint:
            "network": {
                "enabled": NETWORK_AVAILABLE and network_service and network_service.is_initialized,
                "node_id": network_service.network.node_id.to_base58() if (network_service and network_service.is_initialized) else None,
                "peers": len(network_service.node_registry) - 1 if (network_service and network_service.is_initialized) else 0
            },
EOF

# Now let's manually add the network integration with proper indentation
echo "📝 Manually adding network integration..."

# Create a Python script to properly add the integration
cat > fix_main_indentation.py << 'EOF'
#!/usr/bin/env python3
"""Fix main.py with proper network integration."""

import re

# Read the current main.py
with open('backend/main.py', 'r') as f:
    content = f.read()

# Check if network integration already exists
if 'network_integration' in content:
    print("Network integration already exists, fixing indentation...")
    
    # Fix the specific indentation error around line 846
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix any lines that have incorrect indentation for network shutdown
        if 'if NETWORK_AVAILABLE and network_service:' in line:
            # Ensure proper indentation (8 spaces for inside finally block)
            fixed_lines.append('        if NETWORK_AVAILABLE and network_service:')
        elif 'await shutdown_network_service()' in line:
            fixed_lines.append('            await shutdown_network_service()')
        else:
            fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)

else:
    print("Adding network integration...")
    
    # 1. Add imports after WebSocket dependencies
    websocket_import_pattern = r'(# WebSocket dependencies.*?except ImportError:.*?connection_manager = MockConnectionManager\(\))'
    
    network_imports = '''

# Network Integration
try:
    from backend.network_integration import (
        initialize_network_service,
        shutdown_network_service,
        network_service,
        integrate_websocket_with_network
    )
    NETWORK_AVAILABLE = True
except ImportError:
    logger.warning("Network integration not available")
    NETWORK_AVAILABLE = False
    network_service = None'''
    
    content = re.sub(
        websocket_import_pattern,
        r'\1' + network_imports,
        content,
        flags=re.DOTALL
    )
    
    # 2. Add network initialization in lifespan
    component_init_pattern = r'(if COMPONENTS_AVAILABLE:\s+component_registry = await get_component_registry\(\))'
    
    network_init = '''
        
        # Initialize network service
        if NETWORK_AVAILABLE:
            await initialize_network_service(app)
            
            # Integrate WebSocket with network
            if WEBSOCKET_AVAILABLE and network_service and network_service.is_initialized:
                integrate_websocket_with_network(connection_manager, network_service)'''
    
    content = re.sub(
        component_init_pattern,
        r'\1' + network_init,
        content
    )
    
    # 3. Add network shutdown
    shutdown_pattern = r'(logger\.info\("🛑 Shutting down Enhanced CSP Visual Designer Backend"\))'
    
    network_shutdown = '''
        
        # Shutdown network service
        if NETWORK_AVAILABLE and network_service:
            await shutdown_network_service()'''
    
    content = re.sub(
        shutdown_pattern,
        r'\1' + network_shutdown,
        content
    )
    
    # 4. Add network info to root endpoint
    features_pattern = r'("component_registry": COMPONENTS_AVAILABLE,)'
    
    network_feature = '''
            "network": {
                "enabled": NETWORK_AVAILABLE and network_service and network_service.is_initialized,
                "node_id": network_service.network.node_id.to_base58() if (network_service and network_service.is_initialized) else None,
                "peers": len(network_service.node_registry) - 1 if (network_service and network_service.is_initialized) else 0
            },'''
    
    content = re.sub(
        features_pattern,
        r'\1' + network_feature,
        content
    )

# Write the fixed content
with open('backend/main.py', 'w') as f:
    f.write(content)

print("✅ Fixed main.py with proper network integration")
EOF

# Run the fix script
python3 fix_main_indentation.py

# Clean up
rm -f fix_main_indentation.py

echo "✅ Indentation fixed!"
echo ""
echo "Now try running the backend again:"
echo "  python -m backend.main"
