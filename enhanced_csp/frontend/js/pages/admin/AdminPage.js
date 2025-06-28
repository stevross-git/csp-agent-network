// js/pages/admin/AdminPage.js (Complete Updated Implementation)
import {
  recentActivityMock,
  usersMock,
  monitoringDataMock,
  modelsDataMock,
  logsDataMock,
  defaultAgentsMock,
} from "@/utils/apiFallbackData";
class AdminPage extends BaseComponent {
  constructor() {
    super("admin-container", {
      autoInit: true,
      debounceDelay: 300,
    });

    this.components = new Map();
    this.services = new Map();
    this.isInitialized = false;
    this.currentSection = "dashboard";
    this.currentAgentType = "all";

    // Initialize agent data (migrated from existing monolithic code)
    this.agentData = [];
    this.recentActivity = [];
    this.users = [];
    this.monitoringData = [];
    this.modelsData = [];
    this.logsData = [];

    // Real-time update interval
    this.updateInterval = null;
  }

  async loadDependencies() {
    // Check for required dependencies

    if (!window.ApiClient) {
      console.warn("ApiClient not available - API calls will be mocked");
    }

    // Load shared components
    await this.loadSharedComponents();

    // Load page-specific services
    await this.loadServices();

    // Load page-specific components based on priority
    await this.loadComponents();
  }

  async loadSharedComponents() {
    // Load Toast system
    if (!window.toastSystem) {
      await this.loadScript("../js/shared/Toast.js");
    }

    // Load Modal system
    if (!window.Modal) {
      await this.loadScript("../js/shared/Modal.js");
    }

    log_info("Shared components loaded");
  }

  async loadServices() {
    // Initialize agent service (migrated functionality)
    this.agentService = {
      getAll: () => Promise.resolve(this.agentData),
      toggle: (agentId) => this.toggleAgentStatus(agentId),
      create: (agentData) => this.createAgent(agentData),
      delete: (agentId) => this.deleteAgent(agentId),
      duplicate: (agentId) => this.duplicateAgent(agentId),
    };

    this.services.set("agent", this.agentService);
    log_info("Services loaded for admin");
  }

  async loadComponents() {
    try {
      // Load admin-specific components with absolute paths
      await this.loadScript(
        "/js/pages/admin/components/AgentManagement/AgentGrid.js",
      );
      await this.loadScript(
        "/js/pages/admin/components/AgentManagement/AgentCard.js",
      );
      await this.loadScript("/js/pages/admin/components/StatsGrid.js");

      // Initialize components based on current section
      await this.initializeComponents();

      log_info("Components loaded for admin");
    } catch (error) {
      log_error("Failed to load components: " + error.message);
    }
  }

  async initializeComponents() {
    // Initialize navigation component
    this.initializeNavigation();

    // Initialize section-specific components
    await this.initializeSectionComponents(this.currentSection);
  }

  initializeNavigation() {
    const navLinks = document.querySelectorAll(".nav-link");
    navLinks.forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        const section = link
          .getAttribute("onclick")
          ?.match(/showSection\('(.+?)'\)/)?.[1];
        if (section) {
          this.showSection(section);
        }
      });
    });
  }

  async initializeSectionComponents(section) {
    switch (section) {
      case "dashboard":
        await this.initializeDashboard();
        break;
      case "agents":
        await this.initializeAgentManagement();
        break;
      case "monitoring":
        await this.initializeMonitoring();
        break;
      case "users":
        await this.initializeUserManagement();
        break;
      default:
        log_info(`Section ${section} - using default initialization`);
    }
  }

  async initializeDashboard() {
    // Initialize stats grid if available
    const statsContainer = document.getElementById("dashboard-stats");
    if (statsContainer && window.StatsGrid) {
      const statsGrid = new StatsGrid("dashboard-stats", {
        autoRefresh: true,
        refreshInterval: 30000,
        // Callback functions instead of events
        onStatRefresh: (data) => {
          if (data.statId === "active-agents") {
            const activeCount = this.agentData.filter(
              (a) => a.status === "active",
            ).length;
            statsGrid.updateStat("active-agents", { value: activeCount });
          }

          if (data.statId === "total-tasks") {
            const totalTasks = this.agentData.reduce(
              (sum, a) => sum + a.tasksCompleted,
              0,
            );
            statsGrid.updateStat("total-tasks", { value: totalTasks });
          }
        },
        onRefreshAll: () => {
          this.updateStatsGridData(statsGrid);
        },
        onAutoRefresh: () => {
          this.updateStatsGridData(statsGrid);
        },
        onStatClick: (data) => {
          console.log("Stat clicked:", data);
          // Handle stat card clicks here if needed
        },
      });

      this.components.set("dashboardStats", statsGrid);
      await statsGrid.init();

      // Initial data load
      this.updateStatsGridData(statsGrid);
    }

    // Populate tables
    this.populateRecentActivityTable();
    this.updateDashboardStats();
  }

  // Helper method to update stats grid data
  updateStatsGridData(statsGrid) {
    const activeAgents = this.agentData.filter(
      (a) => a.status === "active",
    ).length;
    const totalTasks = this.agentData.reduce(
      (sum, a) => sum + a.tasksCompleted,
      0,
    );
    const errorAgents = this.agentData.filter(
      (a) => a.status === "error",
    ).length;

    statsGrid.updateStats([
      {
        id: "active-agents",
        title: "Active Agents",
        value: activeAgents,
        icon: "🤖",
        color: "var(--success)",
        trend: null,
        description: "Currently running agents",
      },
      {
        id: "total-tasks",
        title: "Tasks Completed",
        value: totalTasks,
        icon: "✅",
        color: "var(--info)",
        trend: null,
        description: "Total tasks processed",
      },
      {
        id: "system-health",
        title: "System Health",
        value:
          errorAgents === 0
            ? "100%"
            : `${Math.round((1 - errorAgents / this.agentData.length) * 100)}%`,
        icon: errorAgents === 0 ? "⚡" : "⚠️",
        color: errorAgents === 0 ? "var(--success)" : "var(--warning)",
        trend: null,
        description: "System health status",
      },
      {
        id: "total-agents",
        title: "Total Agents",
        value: this.agentData.length,
        icon: "🔧",
        color: "var(--primary)",
        trend: null,
        description: "All configured agents",
      },
    ]);
  }

  async initializeAgentManagement() {
    // Initialize AgentGrid
    const agentGridContainer = document.getElementById("agentsGrid");
    if (agentGridContainer && window.AgentGrid) {
      const agentGrid = new AgentGrid("agentsGrid");
      this.components.set("agentGrid", agentGrid);

      // Set up event listeners
      agentGrid.on("agent:toggle", (data) => this.handleAgentToggle(data));
      agentGrid.on("agent:details", (data) => this.handleAgentDetails(data));
      agentGrid.on("agent:logs", (data) => this.handleAgentLogs(data));
      agentGrid.on("agent:duplicate", (data) =>
        this.handleAgentDuplicate(data),
      );
      agentGrid.on("agent:create", () => this.handleAgentCreate());

      await agentGrid.init();
      agentGrid.setAgents(this.agentData);
    }

    // Initialize agent filters
    this.initializeAgentFilters();

    // Initialize agent tabs
    this.initializeAgentTabs();

    this.updateAgentStats();
  }

  async initializeMonitoring() {
    this.populateMonitoringTable();
  }

  async initializeUserManagement() {
    this.populateUsersTable();
  }

  initializeAgentFilters() {
    // Search filter
    const searchInput = document.getElementById("agentSearch");
    if (searchInput) {
      searchInput.addEventListener(
        "input",
        this.debounce(() => {
          const agentGrid = this.components.get("agentGrid");
          if (agentGrid) {
            agentGrid.setSearchTerm(searchInput.value);
          }
        }),
      );
    }

    // Status filter
    const statusFilter = document.getElementById("statusFilter");
    if (statusFilter) {
      statusFilter.addEventListener("change", () => {
        const agentGrid = this.components.get("agentGrid");
        if (agentGrid) {
          agentGrid.setStatusFilter(statusFilter.value);
        }
      });
    }

    // Model filter
    const modelFilter = document.getElementById("modelFilter");
    if (modelFilter) {
      modelFilter.addEventListener("change", () => {
        const agentGrid = this.components.get("agentGrid");
        if (agentGrid) {
          agentGrid.setModelFilter(modelFilter.value);
        }
      });
    }

    // Priority filter
    const priorityFilter = document.getElementById("priorityFilter");
    if (priorityFilter) {
      priorityFilter.addEventListener("change", () => {
        const agentGrid = this.components.get("agentGrid");
        if (agentGrid) {
          agentGrid.setPriorityFilter(priorityFilter.value);
        }
      });
    }
  }

  initializeAgentTabs() {
    const agentTabs = document.querySelectorAll(".agent-tab");
    agentTabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const type = tab.dataset.type;
        this.filterAgentsByType(type);
      });
    });
  }

  render() {
    if (!this.container) {
      log_error("Container not found for admin");
      return;
    }

    // Add page-specific classes
    this.container.classList.add("admin-page", "priority-HIGH");

    // Initialize sample data
    this.loadSampleData();

    // Render page content (existing HTML structure remains)
    this.renderPageContent();

    super.render();
  }

  renderPageContent() {
    // The existing HTML structure from admin.html remains unchanged
    // We're enhancing it with component-based functionality
    log_info(
      "Admin page content rendered - existing HTML structure maintained",
    );
  }

  bindEvents() {
    super.bindEvents();

    // Global navigation events
    this.bindNavigationEvents();

    // Emergency shutdown
    this.bindEmergencyEvents();

    // Form submissions
    this.bindFormEvents();

    // Modal events
    this.bindModalEvents();

    log_info("Events bound for admin");
  }

  bindNavigationEvents() {
    // Navigation toggle
    const navToggle = document.querySelector(".nav-toggle");
    if (navToggle) {
      navToggle.addEventListener("click", () => this.toggleNavigation());
    }

    // Section navigation - already handled in initializeNavigation
  }

  bindEmergencyEvents() {
    const emergencyBtn = document.querySelector(".emergency-btn");
    if (emergencyBtn) {
      emergencyBtn.addEventListener("click", () => this.emergencyShutdown());
    }
  }

  bindFormEvents() {
    const createAgentForm = document.getElementById("createAgentForm");
    if (createAgentForm) {
      createAgentForm.addEventListener("submit", (e) =>
        this.handleCreateAgentSubmit(e),
      );
    }
  }

  bindModalEvents() {
    // Close modals when clicking outside
    document.addEventListener("click", (event) => {
      const modals = document.querySelectorAll(".modal");
      modals.forEach((modal) => {
        if (event.target === modal) {
          modal.classList.remove("active");
        }
      });
    });

    // Close modals on Escape key
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        const activeModals = document.querySelectorAll(".modal.active");
        activeModals.forEach((modal) => {
          modal.classList.remove("active");
        });
      }
    });
  }

  // Navigation methods
  showSection(sectionId) {
    if (this.currentSection === sectionId) return;

    // Hide all sections
    const sections = document.querySelectorAll(".content-section");
    sections.forEach((section) => {
      section.classList.remove("active");
    });

    // Show selected section
    const selectedSection = document.getElementById(sectionId);
    if (selectedSection) {
      selectedSection.classList.add("active");
    }

    // Update navigation
    const navLinks = document.querySelectorAll(".nav-link");
    navLinks.forEach((link) => {
      link.classList.remove("active");
    });

    const activeNavLink = document.querySelector(
      `[onclick="showSection('${sectionId}')"]`,
    );
    if (activeNavLink) {
      activeNavLink.classList.add("active");
    }

    // Initialize section-specific components
    this.initializeSectionComponents(sectionId);

    this.currentSection = sectionId;
  }

  toggleNavigation() {
    const nav = document.getElementById("global-nav");
    if (nav) {
      nav.classList.toggle("collapsed");
    }
  }

  // Agent management methods (migrated from monolithic class)
  handleAgentToggle(data) {
    const agent = this.agentData.find((a) => a.id === data.agentId);
    if (agent) {
      if (agent.status === "active") {
        agent.status = "paused";
        this.showToast(`Agent "${agent.name}" has been paused`, "warning");
      } else {
        agent.status = "active";
        agent.lastActivity = "Just now";
        this.showToast(`Agent "${agent.name}" has been started`, "success");
      }

      this.saveAgentData();
      this.updateAgentGrid();
      this.updateAgentStats();
    }
  }

  handleAgentDetails(data) {
    const agent = this.agentData.find((a) => a.id === data.agentId);
    if (!agent) return;

    const modal = document.getElementById("agentDetailsModal");
    const content = document.getElementById("agentDetailsContent");

    if (modal && content) {
      content.innerHTML = this.renderAgentDetails(agent);
      modal.classList.add("active");
    }
  }

  handleAgentLogs(data) {
    const agent = this.agentData.find((a) => a.id === data.agentId);
    if (!agent) return;

    const modal = document.getElementById("agentDetailsModal");
    const content = document.getElementById("agentDetailsContent");

    if (modal && content) {
      content.innerHTML = this.renderAgentLogs(agent);
      modal.classList.add("active");
    }
  }

  handleAgentDuplicate(data) {
    const agent = this.agentData.find((a) => a.id === data.agentId);
    if (!agent) return;

    const newAgent = {
      ...agent,
      id: `agent-${Date.now()}`,
      name: `${agent.name} (Copy)`,
      status: "stopped",
      tasksCompleted: 0,
      lastActivity: "Never",
      created: new Date().toISOString().split("T")[0],
      logs: [
        {
          time: new Date().toLocaleTimeString(),
          level: "INFO",
          message: "Agent created as duplicate",
        },
      ],
    };

    this.agentData.push(newAgent);
    this.saveAgentData();
    this.updateAgentGrid();
    this.updateAgentStats();

    this.showToast(`Agent "${newAgent.name}" has been created`, "success");
  }

  handleAgentCreate() {
    const modal = document.getElementById("createAgentModal");
    if (modal) {
      modal.classList.add("active");
      const form = document.getElementById("createAgentForm");
      if (form) {
        form.reset();
      }
    }
  }

  handleCreateAgentSubmit(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const capabilities = Array.from(formData.getAll("capabilities"));

    const newAgent = {
      id: `agent-${Date.now()}`,
      name: formData.get("name"),
      type: formData.get("type"),
      status: formData.get("start_immediately") ? "active" : "stopped",
      model: formData.get("model"),
      priority: formData.get("priority"),
      description: formData.get("description") || "No description provided",
      capabilities: capabilities,
      tasksCompleted: 0,
      uptime: "0%",
      lastActivity: formData.get("start_immediately") ? "Just now" : "Never",
      executionMode: "continuous",
      communicationChannel: "internal_api",
      maxTasks: 10,
      timeout: 300,
      autoRestart: true,
      created: new Date().toISOString().split("T")[0],
      logs: [
        {
          time: new Date().toLocaleTimeString(),
          level: "INFO",
          message: `Agent created with ${capabilities.length} capabilities`,
        },
      ],
    };

    this.agentData.push(newAgent);
    this.saveAgentData();
    this.updateAgentGrid();
    this.updateAgentStats();
    this.closeCreateAgentModal();

    this.showToast(
      `Agent "${newAgent.name}" has been created successfully`,
      "success",
    );
  }

  filterAgentsByType(type) {
    this.currentAgentType = type;

    // Update tab appearance
    const tabs = document.querySelectorAll(".agent-tab");
    tabs.forEach((tab) => {
      tab.classList.remove("active");
      if (tab.dataset.type === type) {
        tab.classList.add("active");
      }
    });

    // Update agent grid
    const agentGrid = this.components.get("agentGrid");
    if (agentGrid) {
      agentGrid.setFilter(type);
    }
  }

  deleteAgent(agentId) {
    const agent = this.agentData.find((a) => a.id === agentId);
    if (!agent) return;

    if (
      confirm(
        `Are you sure you want to delete agent "${agent.name}"? This action cannot be undone.`,
      )
    ) {
      this.agentData = this.agentData.filter((a) => a.id !== agentId);
      this.saveAgentData();
      this.updateAgentGrid();
      this.updateAgentStats();
      this.closeAgentDetailsModal();

      this.showToast(`Agent "${agent.name}" has been deleted`, "danger");
    }
  }

  deployAllAgents() {
    const stoppedAgents = this.agentData.filter(
      (a) => a.status === "stopped" || a.status === "paused",
    );

    if (stoppedAgents.length === 0) {
      this.showToast("All agents are already active", "info");
      return;
    }

    stoppedAgents.forEach((agent) => {
      agent.status = "active";
      agent.lastActivity = "Just now";
    });

    this.saveAgentData();
    this.updateAgentGrid();
    this.updateAgentStats();

    this.showToast(
      `${stoppedAgents.length} agents have been deployed`,
      "success",
    );
  }

  refreshAgents() {
    this.updateAgentGrid();
    this.updateAgentStats();
    this.showToast("Agent list refreshed", "info");
  }

  emergencyShutdown() {
    if (
      confirm(
        "⚠️ This will immediately stop all active agents and system processes. Are you sure?",
      )
    ) {
      this.agentData.forEach((agent) => {
        if (agent.status === "active") {
          agent.status = "stopped";
          agent.lastActivity = "Emergency stop";
        }
      });

      this.saveAgentData();
      this.updateAgentGrid();
      this.updateAgentStats();

      this.showToast(
        "🚨 Emergency shutdown executed - all agents stopped",
        "danger",
      );
    }
  }

  // Modal methods
  closeCreateAgentModal() {
    const modal = document.getElementById("createAgentModal");
    if (modal) {
      modal.classList.remove("active");
    }
  }

  closeAgentDetailsModal() {
    const modal = document.getElementById("agentDetailsModal");
    if (modal) {
      modal.classList.remove("active");
    }
  }

  // Rendering methods
  renderAgentDetails(agent) {
    return `
            <div class="agent-details">
                <div class="flex items-center gap-2 mb-3">
                    <h4 style="color: var(--primary); margin: 0;">${agent.name}</h4>
                    <span class="agent-type ${agent.type}">${this.getAgentTypeIcon(agent.type)} ${agent.type}</span>
                    <span class="priority-badge priority-${agent.priority}">${agent.priority}</span>
                </div>
                
                <div class="agent-status mb-3">
                    <span class="status-indicator ${agent.status}"></span>
                    <span style="font-weight: 600;">Status: ${agent.status.charAt(0).toUpperCase() + agent.status.slice(1)}</span>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 2rem;">
                    <div>
                        <strong>Model:</strong> ${agent.model}<br>
                        <strong>Created:</strong> ${agent.created}<br>
                        <strong>Execution Mode:</strong> ${agent.executionMode || "continuous"}<br>
                        <strong>Communication:</strong> ${agent.communicationChannel || "internal_api"}
                    </div>
                    <div>
                        <strong>Tasks Completed:</strong> ${agent.tasksCompleted}<br>
                        <strong>Uptime:</strong> ${agent.uptime}<br>
                        <strong>Max Tasks:</strong> ${agent.maxTasks || 10}<br>
                        <strong>Timeout:</strong> ${agent.timeout || 300}s
                    </div>
                </div>
                
                <div class="mb-3">
                    <strong>Description:</strong><br>
                    <p style="margin-top: 0.5rem; color: var(--text-muted);">${agent.description}</p>
                </div>
                
                <div class="mb-3">
                    <strong>Capabilities:</strong><br>
                    <div style="margin-top: 0.5rem;">
                        ${agent.capabilities
                          .map(
                            (cap) => `
                            <span style="display: inline-block; background: var(--light); padding: 0.25rem 0.5rem; 
                                  border-radius: 12px; font-size: 0.75rem; margin: 0.25rem 0.25rem 0 0;">
                                ${cap.replace("_", " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                            </span>
                        `,
                          )
                          .join("")}
                    </div>
                </div>
                
                <div class="flex gap-2 mt-3">
                    <button class="btn btn-success" onclick="adminPage.handleAgentToggle({agentId: '${agent.id}'}); adminPage.closeAgentDetailsModal();">
                        ${agent.status === "active" ? "⏸️ Pause" : "▶️ Start"}
                    </button>
                    <button class="btn btn-secondary" onclick="adminPage.handleAgentLogs({agentId: '${agent.id}'});">📄 View Logs</button>
                    <button class="btn btn-danger" onclick="adminPage.deleteAgent('${agent.id}')">🗑️ Delete</button>
                </div>
            </div>
        `;
  }

  renderAgentLogs(agent) {
    return `
            <div class="agent-logs">
                <div class="flex justify-between items-center mb-3">
                    <h4 style="color: var(--primary); margin: 0;">Logs for ${agent.name}</h4>
                </div>
                
                <div style="background: #1e293b; color: #e2e8f0; padding: 1rem; border-radius: 8px; 
                            font-family: 'Courier New', monospace; font-size: 0.875rem; max-height: 400px; overflow-y: auto;">
                    ${agent.logs
                      .map(
                        (log) => `
                        <div style="margin-bottom: 0.5rem;">
                            <span style="color: #64748b;">[${log.time}]</span>
                            <span style="color: ${this.getLogLevelColor(log.level)}; font-weight: 600;">${log.level}</span>
                            <span style="margin-left: 1rem;">${log.message}</span>
                        </div>
                    `,
                      )
                      .join("")}
                </div>
            </div>
        `;
  }

  // Utility methods
  getAgentTypeIcon(type) {
    const icons = {
      autonomous: "🧠",
      collaborative: "🤝",
      specialized: "⚡",
      monitoring: "👁️",
    };
    return icons[type] || "🤖";
  }

  getLogLevelColor(level) {
    const colors = {
      INFO: "var(--info)",
      SUCCESS: "var(--success)",
      WARN: "var(--warning)",
      WARNING: "var(--warning)",
      ERROR: "var(--danger)",
      DEBUG: "var(--secondary)",
    };
    return colors[level] || "var(--text)";
  }

  // Data management methods
  loadSampleData() {
    // Load agents from localStorage or use defaults
    const storedAgents = localStorage.getItem("csp_agents");
    if (storedAgents) {
      try {
        this.agentData = JSON.parse(storedAgents);
      } catch (error) {
        console.warn("Failed to load agents from storage, using defaults");
        this.setDefaultAgents();
      }
    } else {
      this.setDefaultAgents();
    }

    // Set other sample data
    this.recentActivity = [...recentActivityMock];
    this.users = [...usersMock];
    this.monitoringData = [...monitoringDataMock];
    this.modelsData = [...modelsDataMock];
    this.logsData = [...logsDataMock];
  }

  setDefaultAgents() {
    this.agentData = [...defaultAgentsMock];

    this.saveAgentData();
  }

  saveAgentData() {
    try {
      localStorage.setItem("csp_agents", JSON.stringify(this.agentData));
    } catch (error) {
      console.warn("Failed to save agents to storage");
    }
  }

  // Update methods
  updateAgentGrid() {
    const agentGrid = this.components.get("agentGrid");
    if (agentGrid) {
      agentGrid.setAgents(this.agentData);
    }
  }

  updateAgentStats() {
    const activeAgents = this.agentData.filter(
      (a) => a.status === "active",
    ).length;
    const pausedAgents = this.agentData.filter(
      (a) => a.status === "paused",
    ).length;
    const totalTasks = this.agentData.reduce(
      (sum, a) => sum + a.tasksCompleted,
      0,
    );

    // Update dashboard stats
    this.updateElementText("active-agents-count", activeAgents);
    this.updateElementText(
      "tasks-completed-count",
      totalTasks.toLocaleString(),
    );

    // Update agent section stats
    this.updateElementText("agent-active-count", activeAgents);
    this.updateElementText("agent-paused-count", pausedAgents);
    this.updateElementText("agent-total-count", this.agentData.length);
    this.updateElementText("agent-tasks-count", totalTasks.toLocaleString());
  }

  updateDashboardStats() {
    this.updateAgentStats();
  }

  updateElementText(id, text) {
    const element = document.getElementById(id);
    if (element) {
      element.textContent = text;
    }
  }

  // Table population methods
  populateRecentActivityTable() {
    const tbody = document.getElementById("recent-activity-tbody");
    if (tbody) {
      tbody.innerHTML = this.recentActivity
        .map(
          (activity) => `
                <tr>
                    <td>${activity.time}</td>
                    <td>${activity.action}</td>
                    <td>${activity.details}</td>
                    <td>${activity.user}</td>
                </tr>
            `,
        )
        .join("");
    }
  }

  populateUsersTable() {
    const tbody = document.getElementById("users-tbody");
    if (tbody) {
      tbody.innerHTML = this.users
        .map(
          (user) => `
                <tr>
                    <td>${user.name}</td>
                    <td>${user.email}</td>
                    <td>${user.role}</td>
                    <td>
                        <span style="color: ${user.status === "Active" ? "var(--success)" : "var(--danger)"};">
                            ${user.status === "Active" ? "🟢" : "🔴"} ${user.status}
                        </span>
                    </td>
                    <td>${user.lastLogin}</td>
                    <td>
                        <button class="btn btn-secondary" style="font-size: 0.75rem; padding: 0.25rem 0.5rem;">Edit</button>
                        <button class="btn btn-danger" style="font-size: 0.75rem; padding: 0.25rem 0.5rem;">Delete</button>
                    </td>
                </tr>
            `,
        )
        .join("");
    }
  }

  populateMonitoringTable() {
    const tbody = document.getElementById("monitoring-tbody");
    if (tbody) {
      tbody.innerHTML = this.monitoringData
        .map(
          (service) => `
                <tr>
                    <td>${service.service}</td>
                    <td>
                        <span style="color: ${service.status === "Running" ? "var(--success)" : "var(--warning)"};">
                            ${service.status === "Running" ? "🟢" : "🟡"} ${service.status}
                        </span>
                    </td>
                    <td>${service.cpu}</td>
                    <td>${service.memory}</td>
                    <td>${service.uptime}</td>
                </tr>
            `,
        )
        .join("");
    }
  }

  // Real-time updates
  startRealTimeUpdates() {
    this.updateInterval = setInterval(() => {
      this.updateLastActivity();
      this.updateAgentStats();
    }, 30000); // Update every 30 seconds
  }

  updateLastActivity() {
    // Simulate real-time activity updates
    this.agentData.forEach((agent) => {
      if (agent.status === "active" && Math.random() < 0.3) {
        const activities = [
          "Just now",
          "1 minute ago",
          "2 minutes ago",
          "3 minutes ago",
          "5 minutes ago",
          "10 minutes ago",
        ];
        agent.lastActivity =
          activities[Math.floor(Math.random() * activities.length)];

        // Occasionally increment task count
        if (Math.random() < 0.1) {
          agent.tasksCompleted++;
        }
      }
    });

    this.saveAgentData();
    this.updateAgentGrid();
  }

  // Utility methods
  showToast(message, type = "info") {
    if (window.toastSystem) {
      window.toastSystem.show(message, type);
    } else {
      // Fallback for basic toast
      console.log(`[${type.toUpperCase()}] ${message}`);
    }
  }

  loadScript(src) {
    return new Promise((resolve, reject) => {
      if (document.querySelector(`script[src="${src}"]`)) {
        resolve();
        return;
      }

      const script = document.createElement("script");
      script.src = src;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  debounce(func, delay = this.options.debounceDelay) {
    let timeoutId;
    return (...args) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
  }

  // Page-specific methods
  async loadPageData() {
    try {
      this.setLoading(true);

      // Load initial data
      this.loadSampleData();

      this.setState({ loaded: true });
    } catch (error) {
      this.showError("Failed to load page data", error);
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
      this.container.classList.add("data-loaded");
    } else {
      this.container.classList.remove("data-loaded");
    }
  }

  onInitialized() {
    super.onInitialized();
    this.isInitialized = true;
    this.loadPageData();
    this.startRealTimeUpdates();

    log_success("AdminPage initialized successfully");
  }

  onDestroy() {
    // Clean up page-specific resources
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }

    this.components.forEach((component) => {
      if (component.destroy) {
        component.destroy();
      }
    });

    this.components.clear();
    this.services.clear();

    super.onDestroy();
  }
}

// Global functions for backwards compatibility with existing HTML onclick handlers
function showSection(sectionId) {
  if (window.adminPage) {
    window.adminPage.showSection(sectionId);
  }
}

function toggleNavigation() {
  if (window.adminPage) {
    window.adminPage.toggleNavigation();
  }
}

function filterAgentsByType(type) {
  if (window.adminPage) {
    window.adminPage.filterAgentsByType(type);
  }
}

function filterAgents() {
  if (window.adminPage) {
    const searchTerm = document.getElementById("agentSearch")?.value || "";
    const agentGrid = window.adminPage.components.get("agentGrid");
    if (agentGrid) {
      agentGrid.setSearchTerm(searchTerm);
    }
  }
}

function openCreateAgentModal() {
  if (window.adminPage) {
    window.adminPage.handleAgentCreate();
  }
}

function closeCreateAgentModal() {
  if (window.adminPage) {
    window.adminPage.closeCreateAgentModal();
  }
}

function closeAgentDetailsModal() {
  if (window.adminPage) {
    window.adminPage.closeAgentDetailsModal();
  }
}

function createAgent(event) {
  if (window.adminPage) {
    window.adminPage.handleCreateAgentSubmit(event);
  }
}

function deployAllAgents() {
  if (window.adminPage) {
    window.adminPage.deployAllAgents();
  }
}

function refreshAgents() {
  if (window.adminPage) {
    window.adminPage.refreshAgents();
  }
}

function emergencyShutdown() {
  if (window.adminPage) {
    window.adminPage.emergencyShutdown();
  }
}

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  // Check if required dependencies are loaded
  if (typeof BaseComponent === "undefined") {
    console.error(
      "BaseComponent not loaded! Include js/shared/BaseComponent.js first.",
    );
    return;
  }

  // Initialize page
  window.adminPage = new AdminPage();
});

// Helper functions for logging in development
function log_info(message) {
  if (console && console.log) {
    console.log(`[AdminPage] ${message}`);
  }
}

function log_error(message) {
  if (console && console.error) {
    console.error(`[AdminPage] ${message}`);
  }
}

function log_success(message) {
  if (console && console.log) {
    console.log(`[AdminPage] ✅ ${message}`);
  }
}
