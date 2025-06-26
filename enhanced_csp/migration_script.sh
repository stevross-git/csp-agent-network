#!/bin/bash

# migrate-page.sh - Website Refactoring Migration Tool
# Usage: ./migrate-page.sh <page-name> [--force]

set -e

PAGE_NAME=$1
FORCE_FLAG=$2

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Validation
if [ -z "$PAGE_NAME" ]; then
    log_error "Usage: ./migrate-page.sh <page-name> [--force]"
    echo ""
    echo "Examples:"
    echo "  ./migrate-page.sh monitoring"
    echo "  ./migrate-page.sh admin --force"
    echo ""
    echo "Available pages to migrate:"
    echo "  🔴 HIGH PRIORITY: admin, monitoring, ai-agents, web_dashboard_ui, developer_tools"
    echo "  🟡 MEDIUM PRIORITY: settings, security, designer, api-explorer, deployment"
    echo "  🟢 LOW PRIORITY: login, index"
    exit 1
fi

# Check if page HTML exists
if [ ! -f "frontend/pages/${PAGE_NAME}.html" ]; then
    log_error "Page frontend/pages/${PAGE_NAME}.html not found!"
    exit 1
fi

# Check if already migrated
if [ -d "frontend/js/pages/${PAGE_NAME}" ] && [ "$FORCE_FLAG" != "--force" ]; then
    log_warning "Page ${PAGE_NAME} appears to already be migrated."
    log_info "Use --force flag to overwrite existing files"
    exit 1
fi

log_info "🔄 Starting migration for ${PAGE_NAME}..."

# Create directory structure
log_info "Creating directory structure..."
mkdir -p "frontend/js/pages/${PAGE_NAME}/components"
mkdir -p "frontend/js/pages/${PAGE_NAME}/services"
mkdir -p "frontend/js/pages/${PAGE_NAME}/utils"
mkdir -p "frontend/css/pages"

# Determine page complexity and priority
get_page_priority() {
    case $PAGE_NAME in
        "admin"|"monitoring"|"ai-agents"|"web_dashboard_ui"|"developer_tools")
            echo "HIGH"
            ;;
        "settings"|"security"|"designer"|"api-explorer"|"deployment")
            echo "MEDIUM"
            ;;
        "login"|"index")
            echo "LOW"
            ;;
        *)
            echo "MEDIUM"
            ;;
    esac
}

PRIORITY=$(get_page_priority)
log_info "Page priority: ${PRIORITY}"

# Create base page class
PAGE_CLASS_NAME="$(echo ${PAGE_NAME} | sed 's/^\(.\)/\U\1/; s/_\(.\)/\U\1/g')Page"
log_info "Creating ${PAGE_CLASS_NAME}..."

cat > "frontend/js/pages/${PAGE_NAME}/${PAGE_CLASS_NAME}.js" << EOF
// js/pages/${PAGE_NAME}/${PAGE_CLASS_NAME}.js
class ${PAGE_CLASS_NAME} extends BaseComponent {
    constructor() {
        super('${PAGE_NAME}-container', {
            autoInit: true,
            debounceDelay: 300
        });
        
        this.components = new Map();
        this.services = new Map();
        this.isInitialized = false;
    }
    
    async loadDependencies() {
        // Initialize API client if not available
        if (!window.ApiClient) {
            throw new Error('ApiClient not available');
        }
        
        // Load page-specific services
        await this.loadServices();
        
        // Load page-specific components based on priority
        await this.loadComponents();
    }
    
    async loadServices() {
        // TODO: Load page-specific services
        // Example:
        // const { ${PAGE_CLASS_NAME}Service } = await import('./services/${PAGE_CLASS_NAME}Service.js');
        // this.services.set('main', new ${PAGE_CLASS_NAME}Service());
        
        log_info("Services loaded for ${PAGE_NAME}");
    }
    
    async loadComponents() {
        // TODO: Load page-specific components based on complexity
        // 
        // For HIGH priority pages (admin, monitoring, etc.):
        // - Load dashboard components
        // - Load data visualization components
        // - Load real-time update components
        //
        // For MEDIUM priority pages:
        // - Load form components
        // - Load basic interaction components
        //
        // For LOW priority pages:
        // - Keep minimal components
        
        log_info("Components loaded for ${PAGE_NAME}");
    }
    
    render() {
        if (!this.container) {
            log_error("Container not found for ${PAGE_NAME}");
            return;
        }
        
        // Add page-specific classes
        this.container.classList.add('${PAGE_NAME}-page', 'priority-${PRIORITY}');
        
        // TODO: Implement page-specific rendering
        this.renderPageContent();
        
        super.render();
    }
    
    renderPageContent() {
        // TODO: Extract existing HTML structure and convert to component-based rendering
        // 
        // 1. Identify reusable sections from existing ${PAGE_NAME}.html
        // 2. Convert to component method calls
        // 3. Implement state-driven updates
        
        const placeholder = this.createElement('div', {
            className: '${PAGE_NAME}-placeholder'
        }, \`
            <h2>${PAGE_NAME} Page (Migrated)</h2>
            <p>This page has been migrated to the new component system.</p>
            <p>Priority: <strong>${PRIORITY}</strong></p>
            <div class="migration-status">
                <span class="status-badge status-migrated">✅ Structure Migrated</span>
                <span class="status-badge status-pending">⏳ Components Pending</span>
                <span class="status-badge status-pending">⏳ Services Pending</span>
            </div>
        \`);
        
        this.container.appendChild(placeholder);
    }
    
    bindEvents() {
        super.bindEvents();
        
        // TODO: Migrate existing event handlers from ${PAGE_NAME}.html
        // 1. Extract inline onclick handlers
        // 2. Convert jQuery event handlers to vanilla JS
        // 3. Implement proper event delegation
        
        this.bindPageSpecificEvents();
    }
    
    bindPageSpecificEvents() {
        // TODO: Add page-specific event bindings
        log_info("Events bound for ${PAGE_NAME}");
    }
    
    // Page-specific methods
    async loadPageData() {
        try {
            this.setLoading(true);
            
            // TODO: Replace direct API calls with service calls
            // const data = await this.services.get('main').fetchData();
            // this.setState({ data });
            
            this.setState({ loaded: true });
        } catch (error) {
            this.showError('Failed to load page data', error);
        } finally {
            this.setLoading(false);
        }
    }
    
    onStateChange(newState, oldState) {
        // Handle state changes and update UI accordingly
        if (newState.loaded !== oldState.loaded) {
            this.updateLoadedState(newState.loaded);
        }
    }
    
    updateLoadedState(isLoaded) {
        if (isLoaded) {
            this.container.classList.add('data-loaded');
        } else {
            this.container.classList.remove('data-loaded');
        }
    }
    
    onInitialized() {
        super.onInitialized();
        this.isInitialized = true;
        this.loadPageData();
        
        log_success("${PAGE_CLASS_NAME} initialized successfully");
    }
    
    onDestroy() {
        // Clean up page-specific resources
        this.components.forEach(component => {
            if (component.destroy) {
                component.destroy();
            }
        });
        
        this.components.clear();
        this.services.clear();
        
        super.onDestroy();
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Check if required dependencies are loaded
    if (typeof BaseComponent === 'undefined') {
        console.error('BaseComponent not loaded! Include js/shared/BaseComponent.js first.');
        return;
    }
    
    if (typeof ApiClient === 'undefined') {
        console.error('ApiClient not loaded! Include js/utils/ApiClient.js first.');
        return;
    }
    
    // Initialize page
    window.${PAGE_CLASS_NAME} = new ${PAGE_CLASS_NAME}();
});

// Helper function for logging in development
function log_info(message) {
    if (console && console.log) {
        console.log(\`[${PAGE_CLASS_NAME}] \${message}\`);
    }
}

function log_error(message) {
    if (console && console.error) {
        console.error(\`[${PAGE_CLASS_NAME}] \${message}\`);
    }
}

function log_success(message) {
    if (console && console.log) {
        console.log(\`[${PAGE_CLASS_NAME}] ✅ \${message}\`);
    }
}
EOF

# Create service template
log_info "Creating service template..."
cat > "frontend/js/pages/${PAGE_NAME}/services/${PAGE_CLASS_NAME}Service.js" << EOF
// js/pages/${PAGE_NAME}/services/${PAGE_CLASS_NAME}Service.js
class ${PAGE_CLASS_NAME}Service {
    constructor(apiClient = window.ApiClient) {
        this.api = apiClient;
        this.cache = new Map();
        this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
    }
    
    async fetchData(useCache = true) {
        const cacheKey = '${PAGE_NAME}_data';
        
        if (useCache && this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheTimeout) {
                return cached.data;
            }
        }
        
        try {
            const response = await this.api.get('/${PAGE_NAME}');
            
            if (response.success) {
                this.cache.set(cacheKey, {
                    data: response.data,
                    timestamp: Date.now()
                });
                return response.data;
            }
            
            throw new Error(response.error || 'Failed to fetch data');
        } catch (error) {
            console.error('${PAGE_CLASS_NAME}Service.fetchData error:', error);
            throw error;
        }
    }
    
    async saveData(data) {
        try {
            const response = await this.api.post('/${PAGE_NAME}', data);
            
            if (response.success) {
                // Invalidate cache
                this.cache.delete('${PAGE_NAME}_data');
                return response.data;
            }
            
            throw new Error(response.error || 'Failed to save data');
        } catch (error) {
            console.error('${PAGE_CLASS_NAME}Service.saveData error:', error);
            throw error;
        }
    }
    
    clearCache() {
        this.cache.clear();
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ${PAGE_CLASS_NAME}Service;
}
EOF

# Create CSS file
log_info "Creating CSS file..."
cat > "frontend/css/pages/${PAGE_NAME}.css" << EOF
/* CSS for ${PAGE_NAME} page */

.${PAGE_NAME}-page {
    min-height: 100vh;
    padding: 20px;
}

.${PAGE_NAME}-page.loading {
    position: relative;
}

.${PAGE_NAME}-page .loading-spinner {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    background: rgba(255, 255, 255, 0.9);
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.${PAGE_NAME}-page .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

/* Priority-specific styling */
.${PAGE_NAME}-page.priority-HIGH {
    border-left: 4px solid #e74c3c;
}

.${PAGE_NAME}-page.priority-MEDIUM {
    border-left: 4px solid #f39c12;
}

.${PAGE_NAME}-page.priority-LOW {
    border-left: 4px solid #27ae60;
}

/* Migration status indicators */
.migration-status {
    display: flex;
    gap: 10px;
    margin-top: 20px;
}

.status-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: bold;
}

.status-migrated {
    background-color: #d4edda;
    color: #155724;
}

.status-pending {
    background-color: #fff3cd;
    color: #856404;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* TODO: Extract styles from existing ${PAGE_NAME}.html */
/* 1. Copy relevant CSS classes from the original page */
/* 2. Organize into logical sections */
/* 3. Remove unused styles */
/* 4. Convert to CSS custom properties for theming */
EOF

# Create migration checklist
log_info "Creating migration checklist..."
cat > "frontend/js/pages/${PAGE_NAME}/MIGRATION_CHECKLIST.md" << EOF
# ${PAGE_NAME} Page Migration Checklist

## ✅ Completed
- [x] Created directory structure
- [x] Created base ${PAGE_CLASS_NAME} class
- [x] Created ${PAGE_CLASS_NAME}Service
- [x] Created base CSS file
- [x] Set up component loading structure

## 📋 TODO

### 1. Extract from Existing HTML (${PAGE_NAME}.html)
- [ ] Identify reusable sections
- [ ] Extract inline JavaScript
- [ ] Convert jQuery to vanilla JS
- [ ] Identify API endpoints used
- [ ] Extract CSS classes and styles

### 2. Component Creation
- [ ] Create main dashboard component (if applicable)
- [ ] Create form components
- [ ] Create data display components
- [ ] Create navigation/menu components
- [ ] Create modal/dialog components

### 3. Service Implementation
- [ ] Implement actual API endpoints
- [ ] Add error handling
- [ ] Add caching strategy
- [ ] Add real-time updates (WebSocket if needed)

### 4. Testing
- [ ] Unit tests for components
- [ ] Integration tests for services
- [ ] E2E tests for critical paths
- [ ] Performance testing

### 5. Optimization
- [ ] Lazy loading implementation
- [ ] Code splitting
- [ ] Bundle size analysis
- [ ] Performance monitoring

## 🎯 Priority: ${PRIORITY}

### High Priority Actions (Do First):
$(case $PRIORITY in
    "HIGH")
        echo "- Focus on complex interactive components"
        echo "- Implement real-time data updates"
        echo "- Add comprehensive error handling"
        echo "- Create dashboard-style components"
        ;;
    "MEDIUM")
        echo "- Focus on form handling and validation"
        echo "- Implement basic API integration"
        echo "- Create standard UI components"
        ;;
    "LOW")
        echo "- Basic migration to new structure"
        echo "- Simple component extraction"
        echo "- Keep minimal complexity"
        ;;
esac)

## 📝 Notes
- Original file: frontend/pages/${PAGE_NAME}.html
- Dependencies: BaseComponent, ApiClient
- Component count estimate: $(case $PRIORITY in "HIGH") echo "5-10";; "MEDIUM") echo "3-5";; "LOW") echo "1-3";; esac)
- Estimated effort: $(case $PRIORITY in "HIGH") echo "2-3 weeks";; "MEDIUM") echo "1-2 weeks";; "LOW") echo "3-5 days";; esac)

## 🔄 Migration Strategy
1. Keep original ${PAGE_NAME}.html functional during migration
2. Gradually move functionality to new component system
3. Test thoroughly before removing old code
4. Update navigation links when ready
EOF

# Create updated HTML file that uses the new system
log_info "Creating updated HTML file..."
cp "frontend/pages/${PAGE_NAME}.html" "frontend/pages/${PAGE_NAME}.html.backup"

cat > "frontend/pages/${PAGE_NAME}.html" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>$(echo ${PAGE_NAME} | sed 's/^\(.\)/\U\1/; s/_/ /g') - Enhanced CSP System</title>
    
    <!-- Shared Styles -->
    <link rel="stylesheet" href="../css/shared/base.css">
    <link rel="stylesheet" href="../css/shared/components.css">
    <link rel="stylesheet" href="../css/shared/theme.css">
    
    <!-- Page-specific styles -->
    <link rel="stylesheet" href="../css/pages/${PAGE_NAME}.css">
</head>
<body>
    <!-- Auth Guard -->
    <div id="auth-guard"></div>
    
    <!-- Global Header -->
    <header id="global-header"></header>
    
    <!-- Global Navigation -->
    <nav id="global-nav"></nav>
    
    <!-- Page Content -->
    <main id="${PAGE_NAME}-container" class="${PAGE_NAME}-page">
        <!-- New component system will render here -->
        <div class="migration-notice">
            <h1>$(echo ${PAGE_NAME} | sed 's/^\(.\)/\U\1/; s/_/ /g') Page</h1>
            <p>This page has been migrated to the new component architecture.</p>
        </div>
    </main>
    
    <!-- Global Footer -->
    <footer id="global-footer"></footer>
    
    <!-- Modal Container -->
    <div id="modal-container"></div>
    
    <!-- Toast Container -->
    <div id="toast-container"></div>
    
    <!-- Shared Scripts -->
    <script src="../js/shared/BaseComponent.js"></script>
    <script src="../js/utils/ApiClient.js"></script>
    
    <!-- Page-specific scripts -->
    <script src="../js/pages/${PAGE_NAME}/${PAGE_CLASS_NAME}.js"></script>
</body>
</html>
EOF

# Generate summary report
log_success "Migration structure created for ${PAGE_NAME}!"
echo ""
echo "📊 Migration Summary:"
echo "   📁 Created: frontend/js/pages/${PAGE_NAME}/"
echo "   📄 Created: ${PAGE_CLASS_NAME}.js"
echo "   🔧 Created: ${PAGE_CLASS_NAME}Service.js"
echo "   🎨 Created: ${PAGE_NAME}.css"
echo "   📝 Created: MIGRATION_CHECKLIST.md"
echo "   🔄 Updated: ${PAGE_NAME}.html"
echo "   💾 Backup: ${PAGE_NAME}.html.backup"
echo ""
echo "🎯 Next Steps:"
echo "   1. Review MIGRATION_CHECKLIST.md"
echo "   2. Extract components from original ${PAGE_NAME}.html.backup"
echo "   3. Implement page-specific functionality"
echo "   4. Test the migrated page"
echo ""
echo "📚 Documentation: See frontend/js/pages/${PAGE_NAME}/MIGRATION_CHECKLIST.md"

log_success "Migration for ${PAGE_NAME} completed! 🎉"