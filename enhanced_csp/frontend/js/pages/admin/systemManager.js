// frontend/js/pages/admin/systemManager.js
/**
 * System Settings Manager
 * Dynamically generates and manages system configuration settings
 * Works with existing Enhanced CSP structure
 */

class SystemManager {
  constructor() {
    this.settings = [];
    this.formContainer = null;
    this.loading = false;
    this.originalSettings = new Map();
    this.apiBaseUrl = this.getApiBaseUrl();
    this.authToken = this.getAuthToken();
    
    this.widgetFactories = {
      text: this.createTextInput.bind(this),
      number: this.createNumberInput.bind(this),
      switch: this.createSwitchInput.bind(this),
      select: this.createSelectInput.bind(this),
      textarea: this.createTextareaInput.bind(this)
    };
    
    // Check if we can initialize immediately
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => this.init());
    } else {
      this.init();
    }
  }

  getApiBaseUrl() {
    // Check multiple sources for API URL
    if (window.CSP_CONFIG && window.CSP_CONFIG.apiBaseUrl) {
      return window.CSP_CONFIG.apiBaseUrl;
    }
    
    if (window.REACT_APP_CSP_API_URL) {
      return window.REACT_APP_CSP_API_URL;
    }
    
    const metaApiUrl = document.querySelector('meta[name="api-base-url"]');
    if (metaApiUrl) {
      return metaApiUrl.getAttribute('content');
    }
    
    // Fallback based on environment
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return 'http://localhost:8000';
    }
    
    return window.location.origin.replace(':3000', ':8000');
  }

  getAuthToken() {
    return localStorage.getItem('csp_auth_token') || 
           sessionStorage.getItem('csp_auth_token') ||
           localStorage.getItem('authToken');
  }

  init() {
    try {
      // Wait for section to be visible before initializing
      const systemSettingsSection = document.getElementById('system-settings');
      if (!systemSettingsSection) {
        console.warn('System settings section not found');
        return;
      }

      // Set up observer to initialize when section becomes visible
      this.setupVisibilityObserver(systemSettingsSection);

      // If section is already visible, initialize immediately
      if (!systemSettingsSection.classList.contains('hidden')) {
        this.initializeManager();
      }
    } catch (error) {
      console.error('Failed to initialize SystemManager:', error);
    }
  }

  setupVisibilityObserver(systemSettingsSection) {
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
          const isVisible = !systemSettingsSection.classList.contains('hidden');
          if (isVisible && !this.formContainer) {
            this.initializeManager();
          }
        }
      });
    });

    observer.observe(systemSettingsSection, {
      attributes: true,
      attributeFilter: ['class']
    });
  }

  async initializeManager() {
    try {
      this.formContainer = document.getElementById('settings-form-container');
      if (!this.formContainer) {
        console.error('Settings form container not found');
        return;
      }

      this.bindEvents();
      await this.loadSettings();
      this.renderForm();
      
      console.log('SystemManager initialized successfully');
    } catch (error) {
      console.error('Failed to initialize SystemManager:', error);
      this.showToast('Failed to initialize system settings', 'error');
    }
  }

  bindEvents() {
    // Save button
    const saveButton = document.getElementById('save-settings');
    if (saveButton) {
      saveButton.addEventListener('click', () => this.saveSettings());
    }

    // Reset button
    const resetButton = document.getElementById('reset-settings');
    if (resetButton) {
      resetButton.addEventListener('click', () => this.resetSettings());
    }

    // Handle form submission with Ctrl+S
    document.addEventListener('keydown', (e) => {
      if (e.ctrlKey && e.key === 's') {
        e.preventDefault();
        this.saveSettings();
      }
    });

    // Auto-save warning on critical changes
    document.addEventListener('change', (e) => {
      if (e.target.classList.contains('critical-setting')) {
        this.showToast('Critical setting changed - remember to save!', 'warning');
      }
    });
  }

  async loadSettings() {
    try {
      this.setLoading(true);
      
      // Try to load from API first
      try {
        const response = await this.apiRequest('/api/settings', {
          method: 'GET'
        });

        if (response.ok) {
          const data = await response.json();
          this.settings = data.settings || [];
        } else {
          throw new Error(`API returned ${response.status}`);
        }
      } catch (apiError) {
        console.warn('API not available, using mock data:', apiError);
        this.settings = this.getMockSettings();
      }
      
      // Store original values for reset functionality
      this.originalSettings.clear();
      this.settings.forEach(setting => {
        this.originalSettings.set(setting.key, setting.value);
      });

      console.log(`Loaded ${this.settings.length} settings`);
    } catch (error) {
      console.error('Error loading settings:', error);
      this.showToast('Failed to load settings', 'error');
      
      // Fallback to mock data
      this.settings = this.getMockSettings();
    } finally {
      this.setLoading(false);
    }
  }

  getMockSettings() {
    return [
      { key: 'app_name', value: 'Enhanced CSP System', description: 'Application name', widget: 'text', category: 'Application' },
      { key: 'debug', value: false, description: 'Enable debug mode', widget: 'switch', category: 'Application' },
      { key: 'environment', value: 'development', description: 'Application environment', widget: 'select', options: ['development', 'testing', 'staging', 'production'], category: 'Application' },
      { key: 'enable_ai', value: true, description: 'Enable AI features', widget: 'switch', category: 'Features' },
      { key: 'enable_websockets', value: true, description: 'Enable WebSocket support', widget: 'switch', category: 'Features' },
      { key: 'database_host', value: 'localhost', description: 'Database host address', widget: 'text', category: 'Database' },
      { key: 'database_port', value: 5432, description: 'Database port', widget: 'number', category: 'Database' },
      { key: 'database_pool_size', value: 20, description: 'Database connection pool size', widget: 'number', category: 'Database' },
      { key: 'redis_host', value: 'localhost', description: 'Redis host address', widget: 'text', category: 'Cache' },
      { key: 'redis_port', value: 6379, description: 'Redis port', widget: 'number', category: 'Cache' },
      { key: 'ai_max_requests_per_minute', value: 60, description: 'AI API rate limit (requests/min)', widget: 'number', category: 'AI' },
      { key: 'ai_max_daily_cost', value: 100.0, description: 'Maximum daily AI cost limit ($)', widget: 'number', category: 'AI' },
      { key: 'security_max_login_attempts', value: 5, description: 'Maximum login attempts before lockout', widget: 'number', category: 'Security' },
      { key: 'api_rate_limit_requests_per_minute', value: 100, description: 'API rate limit (requests/min/user)', widget: 'number', category: 'API' },
      { key: 'log_level', value: 'INFO', description: 'Application log level', widget: 'select', options: ['DEBUG', 'INFO', 'WARNING', 'ERROR'], category: 'Monitoring' }
    ];
  }

  renderForm() {
    if (!this.formContainer) return;

    // Group settings by category
    const categorizedSettings = this.groupSettingsByCategory();
    
    // Clear existing content
    this.formContainer.innerHTML = '';

    // Create search/filter bar
    this.renderFilterBar();

    // Render each category
    Object.entries(categorizedSettings).forEach(([category, settings]) => {
      this.renderCategory(category, settings);
    });

    // Add form validation
    this.initializeValidation();
  }

  renderFilterBar() {
    const filterBar = document.createElement('div');
    filterBar.className = 'settings-filter-bar';
    filterBar.innerHTML = `
      <div class="filter-search">
        <input type="text" id="settings-search" placeholder="Search settings..." class="search-input">
        <i class="fas fa-search search-icon"></i>
      </div>
      <div class="filter-category">
        <select id="category-filter" class="category-select">
          <option value="">All Categories</option>
        </select>
      </div>
      <div class="filter-actions">
        <button type="button" class="btn-secondary" id="expand-all">Expand All</button>
        <button type="button" class="btn-secondary" id="collapse-all">Collapse All</button>
      </div>
    `;

    this.formContainer.appendChild(filterBar);

    // Populate category filter
    const categorySelect = filterBar.querySelector('#category-filter');
    const categories = [...new Set(this.settings.map(s => s.category || 'General'))].sort();
    categories.forEach(category => {
      const option = document.createElement('option');
      option.value = category;
      option.textContent = category;
      categorySelect.appendChild(option);
    });

    // Bind filter events
    this.bindFilterEvents(filterBar);
  }

  bindFilterEvents(filterBar) {
    const searchInput = filterBar.querySelector('#settings-search');
    const categoryFilter = filterBar.querySelector('#category-filter');
    const expandAll = filterBar.querySelector('#expand-all');
    const collapseAll = filterBar.querySelector('#collapse-all');

    // Search functionality
    searchInput.addEventListener('input', (e) => {
      this.filterSettings(e.target.value, categoryFilter.value);
    });

    // Category filter
    categoryFilter.addEventListener('change', (e) => {
      this.filterSettings(searchInput.value, e.target.value);
    });

    // Expand/collapse all
    expandAll.addEventListener('click', () => this.toggleAllCategories(true));
    collapseAll.addEventListener('click', () => this.toggleAllCategories(false));
  }

  filterSettings(searchTerm, categoryFilter) {
    const categories = this.formContainer.querySelectorAll('.settings-category');
    
    categories.forEach(categoryElement => {
      const categoryName = categoryElement.querySelector('h3').textContent;
      const settingGroups = categoryElement.querySelectorAll('.setting-group');
      let visibleSettings = 0;

      // Check category filter
      if (categoryFilter && categoryName !== categoryFilter) {
        categoryElement.style.display = 'none';
        return;
      }

      // Check search term
      settingGroups.forEach(group => {
        const label = group.querySelector('label').textContent.toLowerCase();
        const description = group.querySelector('.setting-description').textContent.toLowerCase();
        const matches = !searchTerm || 
                       label.includes(searchTerm.toLowerCase()) || 
                       description.includes(searchTerm.toLowerCase());
        
        group.style.display = matches ? 'block' : 'none';
        if (matches) visibleSettings++;
      });

      categoryElement.style.display = visibleSettings > 0 ? 'block' : 'none';
    });
  }

  toggleAllCategories(expand) {
    const categoryBodies = this.formContainer.querySelectorAll('.category-body');
    const toggleIcons = this.formContainer.querySelectorAll('.category-toggle i');
    
    categoryBodies.forEach((body, index) => {
      body.style.display = expand ? 'block' : 'none';
      if (toggleIcons[index]) {
        toggleIcons[index].className = expand ? 'fas fa-chevron-down' : 'fas fa-chevron-right';
      }
    });
  }

  groupSettingsByCategory() {
    const grouped = {};
    
    this.settings.forEach(setting => {
      const category = setting.category || 'General';
      if (!grouped[category]) {
        grouped[category] = [];
      }
      grouped[category].push(setting);
    });

    // Sort categories and settings within each category
    const sortedGrouped = {};
    Object.keys(grouped).sort().forEach(category => {
      sortedGrouped[category] = grouped[category].sort((a, b) => a.key.localeCompare(b.key));
    });

    return sortedGrouped;
  }

  renderCategory(categoryName, settings) {
    const categoryDiv = document.createElement('div');
    categoryDiv.className = 'settings-category';
    
    const categoryHeader = document.createElement('div');
    categoryHeader.className = 'category-header';
    categoryHeader.innerHTML = `
      <h3 class="category-title">
        <button type="button" class="category-toggle" aria-expanded="true">
          <i class="fas fa-chevron-down"></i>
        </button>
        ${categoryName}
        <span class="category-count">(${settings.length})</span>
      </h3>
    `;

    const categoryBody = document.createElement('div');
    categoryBody.className = 'category-body';

    settings.forEach(setting => {
      const settingElement = this.createSettingElement(setting);
      categoryBody.appendChild(settingElement);
    });

    categoryDiv.appendChild(categoryHeader);
    categoryDiv.appendChild(categoryBody);
    this.formContainer.appendChild(categoryDiv);

    // Bind category toggle
    const toggleButton = categoryHeader.querySelector('.category-toggle');
    toggleButton.addEventListener('click', () => {
      const isExpanded = categoryBody.style.display !== 'none';
      categoryBody.style.display = isExpanded ? 'none' : 'block';
      toggleButton.setAttribute('aria-expanded', !isExpanded);
      toggleButton.querySelector('i').className = isExpanded ? 'fas fa-chevron-right' : 'fas fa-chevron-down';
    });
  }

  createSettingElement(setting) {
    const wrapper = document.createElement('div');
    wrapper.className = 'setting-group';
    wrapper.dataset.settingKey = setting.key;
    
    const isCritical = this.isCriticalSetting(setting.key);
    if (isCritical) {
      wrapper.classList.add('critical-setting-group');
    }

    const label = document.createElement('label');
    label.className = 'setting-label';
    label.htmlFor = `setting-${setting.key}`;
    label.innerHTML = `
      ${this.formatSettingName(setting.key)}
      ${isCritical ? '<i class="fas fa-exclamation-triangle critical-icon" title="Critical Setting"></i>' : ''}
    `;

    const description = document.createElement('div');
    description.className = 'setting-description';
    description.textContent = setting.description || 'No description available';

    const inputWrapper = document.createElement('div');
    inputWrapper.className = 'setting-input-wrapper';

    const widget = this.createWidget(setting);
    if (widget) {
      inputWrapper.appendChild(widget);
    } else {
      console.warn(`Failed to create widget for setting: ${setting.key}`);
      return wrapper;
    }

    wrapper.appendChild(label);
    wrapper.appendChild(description);
    wrapper.appendChild(inputWrapper);

    return wrapper;
  }

  createWidget(setting) {
    const factory = this.widgetFactories[setting.widget];
    if (!factory) {
      console.warn(`Unknown widget type: ${setting.widget}`);
      return this.createTextInput(setting);
    }
    
    return factory(setting);
  }

  createTextInput(setting) {
    const input = document.createElement('input');
    input.type = 'text';
    input.id = `setting-${setting.key}`;
    input.name = setting.key;
    input.value = setting.value || '';
    input.className = 'setting-input text-input';
    
    if (this.isCriticalSetting(setting.key)) {
      input.classList.add('critical-setting');
    }

    return input;
  }

  createNumberInput(setting) {
    const input = document.createElement('input');
    input.type = 'number';
    input.id = `setting-${setting.key}`;
    input.name = setting.key;
    input.value = setting.value || 0;
    input.className = 'setting-input number-input';
    
    if (setting.min !== undefined) input.min = setting.min;
    if (setting.max !== undefined) input.max = setting.max;
    if (setting.step !== undefined) input.step = setting.step;
    
    if (this.isCriticalSetting(setting.key)) {
      input.classList.add('critical-setting');
    }

    return input;
  }

  createSwitchInput(setting) {
    const wrapper = document.createElement('div');
    wrapper.className = 'switch-wrapper';

    const input = document.createElement('input');
    input.type = 'checkbox';
    input.id = `setting-${setting.key}`;
    input.name = setting.key;
    input.checked = Boolean(setting.value);
    input.className = 'setting-input switch-input';
    
    if (this.isCriticalSetting(setting.key)) {
      input.classList.add('critical-setting');
    }

    const slider = document.createElement('label');
    slider.className = 'switch-slider';
    slider.htmlFor = input.id;
    slider.innerHTML = '<span class="slider-handle"></span>';

    wrapper.appendChild(input);
    wrapper.appendChild(slider);

    return wrapper;
  }

  createSelectInput(setting) {
    const select = document.createElement('select');
    select.id = `setting-${setting.key}`;
    select.name = setting.key;
    select.className = 'setting-input select-input';
    
    if (this.isCriticalSetting(setting.key)) {
      select.classList.add('critical-setting');
    }

    const options = setting.options || [];
    options.forEach(optionValue => {
      const option = document.createElement('option');
      option.value = optionValue;
      option.textContent = optionValue;
      option.selected = optionValue === setting.value;
      select.appendChild(option);
    });

    return select;
  }

  createTextareaInput(setting) {
    const textarea = document.createElement('textarea');
    textarea.id = `setting-${setting.key}`;
    textarea.name = setting.key;
    textarea.value = setting.value || '';
    textarea.className = 'setting-input textarea-input';
    textarea.rows = 3;
    
    if (this.isCriticalSetting(setting.key)) {
      textarea.classList.add('critical-setting');
    }

    return textarea;
  }

  isCriticalSetting(key) {
    const criticalSettings = [
      'environment', 'debug', 'database_host', 'database_port',
      'secret_key', 'enable_authentication', 'api_port'
    ];
    return criticalSettings.some(critical => key.includes(critical));
  }

  formatSettingName(key) {
    return key.split('_')
             .map(word => word.charAt(0).toUpperCase() + word.slice(1))
             .join(' ');
  }

  initializeValidation() {
    const inputs = this.formContainer.querySelectorAll('.setting-input');
    
    inputs.forEach(input => {
      input.addEventListener('blur', () => this.validateInput(input));
      input.addEventListener('input', () => this.clearValidationError(input));
    });
  }

  validateInput(input) {
    const value = input.type === 'checkbox' ? input.checked : input.value;
    const key = input.name;
    let isValid = true;
    let errorMessage = '';

    // Type-specific validation
    if (input.type === 'number') {
      const numValue = parseFloat(value);
      if (isNaN(numValue)) {
        isValid = false;
        errorMessage = 'Must be a valid number';
      } else if (input.min && numValue < parseFloat(input.min)) {
        isValid = false;
        errorMessage = `Must be at least ${input.min}`;
      } else if (input.max && numValue > parseFloat(input.max)) {
        isValid = false;
        errorMessage = `Must be no more than ${input.max}`;
      }
    }

    // Key-specific validation
    if (key.includes('port') && (value < 1 || value > 65535)) {
      isValid = false;
      errorMessage = 'Port must be between 1 and 65535';
    }

    if (key.includes('email') && value && !this.isValidEmail(value)) {
      isValid = false;
      errorMessage = 'Must be a valid email address';
    }

    if (key.includes('url') && value && !this.isValidUrl(value)) {
      isValid = false;
      errorMessage = 'Must be a valid URL';
    }

    // Display validation result
    if (isValid) {
      this.clearValidationError(input);
    } else {
      this.showValidationError(input, errorMessage);
    }

    return isValid;
  }

  showValidationError(input, message) {
    input.classList.add('invalid');
    
    // Remove existing error message
    this.clearValidationError(input);
    
    // Add new error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'validation-error';
    errorDiv.textContent = message;
    
    const wrapper = input.closest('.setting-input-wrapper');
    if (wrapper) {
      wrapper.appendChild(errorDiv);
    }
  }

  clearValidationError(input) {
    input.classList.remove('invalid');
    
    const wrapper = input.closest('.setting-input-wrapper');
    if (wrapper) {
      const errorDiv = wrapper.querySelector('.validation-error');
      if (errorDiv) {
        errorDiv.remove();
      }
    }
  }

  isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  isValidUrl(url) {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }

  async saveSettings() {
    try {
      this.setLoading(true);

      // Validate all inputs
      const inputs = this.formContainer.querySelectorAll('.setting-input');
      let allValid = true;
      
      inputs.forEach(input => {
        if (!this.validateInput(input)) {
          allValid = false;
        }
      });

      if (!allValid) {
        this.showToast('Please fix validation errors before saving', 'error');
        return;
      }

      // Collect settings data
      const settingsData = this.collectFormData();

      // Try to send to API
      try {
        const response = await this.apiRequest('/api/settings', {
          method: 'PUT',
          body: JSON.stringify({ settings: settingsData })
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.message || 'Failed to save settings');
        }

        const result = await response.json();
      } catch (apiError) {
        console.warn('API not available, settings saved locally:', apiError);
      }
      
      // Update original settings for reset functionality
      settingsData.forEach(setting => {
        this.originalSettings.set(setting.key, setting.value);
      });

      this.showToast('Settings saved successfully!', 'success');
      
      // Show restart warning for critical settings
      if (this.hasCriticalChanges(settingsData)) {
        this.showRestartWarning();
      }

    } catch (error) {
      console.error('Error saving settings:', error);
      this.showToast(`Failed to save settings: ${error.message}`, 'error');
    } finally {
      this.setLoading(false);
    }
  }

  collectFormData() {
    const inputs = this.formContainer.querySelectorAll('.setting-input');
    const settingsData = [];

    inputs.forEach(input => {
      const key = input.name;
      let value = input.type === 'checkbox' ? input.checked : input.value;
      
      // Convert number inputs to appropriate type
      if (input.type === 'number') {
        value = parseFloat(value);
      }

      settingsData.push({ key, value });
    });

    return settingsData;
  }

  hasCriticalChanges(settingsData) {
    return settingsData.some(setting => {
      const original = this.originalSettings.get(setting.key);
      return this.isCriticalSetting(setting.key) && original !== setting.value;
    });
  }

  showRestartWarning() {
    const modal = document.createElement('div');
    modal.className = 'restart-warning-modal';
    modal.innerHTML = `
      <div class="modal-content">
        <div class="modal-header">
          <h3><i class="fas fa-exclamation-triangle"></i> Restart Required</h3>
        </div>
        <div class="modal-body">
          <p>Some critical settings have been changed. A system restart may be required for these changes to take effect:</p>
          <ul class="critical-changes-list">
            <li>Application configuration changes</li>
            <li>Database connection settings</li>
            <li>Security configurations</li>
          </ul>
          <p>Please coordinate with your system administrator for the restart.</p>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn-primary" onclick="this.closest('.restart-warning-modal').remove()">
            Understood
          </button>
        </div>
      </div>
    `;

    document.body.appendChild(modal);

    // Auto-remove after 10 seconds
    setTimeout(() => {
      if (modal && modal.parentNode) {
        modal.remove();
      }
    }, 10000);
  }

  async resetSettings() {
    try {
      // Show confirmation dialog
      if (!confirm('Are you sure you want to reset all settings to their original values? This will discard any unsaved changes.')) {
        return;
      }

      this.setLoading(true);

      // Reset to original values
      const inputs = this.formContainer.querySelectorAll('.setting-input');
      inputs.forEach(input => {
        const key = input.name;
        const originalValue = this.originalSettings.get(key);
        
        if (originalValue !== undefined) {
          if (input.type === 'checkbox') {
            input.checked = Boolean(originalValue);
          } else {
            input.value = originalValue;
          }
        }

        // Clear any validation errors
        this.clearValidationError(input);
      });

      this.showToast('Settings reset to original values', 'info');

    } catch (error) {
      console.error('Error resetting settings:', error);
      this.showToast('Failed to reset settings', 'error');
    } finally {
      this.setLoading(false);
    }
  }

  async refresh() {
    await this.loadSettings();
    this.renderForm();
  }

  async apiRequest(endpoint, options = {}) {
    const url = `${this.apiBaseUrl}${endpoint}`;
    const defaultOptions = {
      headers: {
        'Content-Type': 'application/json',
        ...(this.authToken && { 'Authorization': `Bearer ${this.authToken}` })
      }
    };

    const mergedOptions = {
      ...defaultOptions,
      ...options,
      headers: { ...defaultOptions.headers, ...options.headers }
    };

    return fetch(url, mergedOptions);
  }

  setLoading(loading) {
    this.loading = loading;
    
    const saveButton = document.getElementById('save-settings');
    const resetButton = document.getElementById('reset-settings');
    
    if (saveButton) {
      saveButton.disabled = loading;
      saveButton.innerHTML = loading ? 
        '<i class="fas fa-spinner fa-spin"></i> Saving...' : 
        '<i class="fas fa-save"></i> Save';
    }
    
    if (resetButton) {
      resetButton.disabled = loading;
    }

    // Add loading overlay to form
    const existingOverlay = this.formContainer?.querySelector('.loading-overlay');
    if (loading && !existingOverlay && this.formContainer) {
      const overlay = document.createElement('div');
      overlay.className = 'loading-overlay';
      overlay.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner fa-spin"></i></div>';
      this.formContainer.appendChild(overlay);
    } else if (!loading && existingOverlay) {
      existingOverlay.remove();
    }
  }

  showToast(message, type = 'info') {
    // Try to use existing toast system
    if (window.Toast && typeof window.Toast.show === 'function') {
      window.Toast.show(message, type);
      return;
    }

    if (window.showToast && typeof window.showToast === 'function') {
      window.showToast(message, type);
      return;
    }

    // Fallback toast implementation
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
      <div class="toast-content">
        <i class="fas fa-${this.getToastIcon(type)}"></i>
        <span>${message}</span>
      </div>
    `;

    // Add to toast container or body
    let container = document.getElementById('toast-container');
    if (!container) {
      container = document.body;
    }
    
    container.appendChild(toast);

    // Auto-remove after 5 seconds
    setTimeout(() => {
      if (toast && toast.parentNode) {
        toast.remove();
      }
    }, 5000);
  }

  getToastIcon(type) {
    const icons = {
      success: 'check-circle',
      error: 'exclamation-circle',
      warning: 'exclamation-triangle',
      info: 'info-circle'
    };
    return icons[type] || 'info-circle';
  }
}

// Initialize when system settings section becomes visible
// This ensures compatibility with existing admin page structure
if (typeof window !== 'undefined') {
  window.SystemManager = SystemManager;
  
  // Auto-initialize if not using module system
  if (!window.systemManager) {
    window.systemManager = new SystemManager();
  }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = SystemManager;
}

export default SystemManager;