# File: cli/manage.py
"""
CSP Visual Designer Management CLI
=================================
Command-line interface for managing the CSP Visual Designer backend
"""

import asyncio
import click
import json
import sys
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import uvicorn
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.json import JSON
import yaml

# Import backend components
from backend.config.settings import get_settings, reload_configuration
from backend.database.connection import startup_database, shutdown_database, check_database_health
from backend.database.migrate import DatabaseMigrator
from backend.auth.auth_system import create_initial_admin
from backend.components.registry import get_component_registry
from backend.ai.ai_integration import get_usage_statistics, reset_usage_statistics
from backend.monitoring.performance import get_performance_monitor

# Setup rich console
console = Console()

# ============================================================================
# CLI GROUPS
# ============================================================================

@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def cli(config, verbose):
    """CSP Visual Designer Management CLI"""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    if config:
        os.environ['CONFIG_FILE'] = config
        reload_configuration()

# ============================================================================
# SERVER MANAGEMENT
# ============================================================================

@cli.group()
def server():
    """Server management commands"""
    pass

@server.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--workers', default=1, help='Number of worker processes')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.option('--log-level', default='info', help='Log level')
def start(host, port, workers, reload, log_level):
    """Start the server"""
    console.print(f"🚀 Starting CSP Visual Designer API server on {host}:{port}")
    
    if reload:
        console.print("⚠️  Auto-reload enabled (development mode)")
    
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level=log_level,
        access_log=True
    )

@server.command()
def status():
    """Check server status"""
    import httpx
    
    settings = get_settings()
    url = f"http://{settings.api.host}:{settings.api.port}/health"
    
    try:
        with console.status("Checking server status..."):
            response = httpx.get(url, timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            console.print("✅ Server is running and healthy")
            
            table = Table(title="Server Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            
            for component, info in health_data.get("components", {}).items():
                status_text = info.get("status", "unknown")
                table.add_row(component, status_text)
            
            console.print(table)
        else:
            console.print(f"❌ Server returned status code: {response.status_code}")
    
    except Exception as e:
        console.print(f"❌ Failed to connect to server: {e}")

# ============================================================================
# DATABASE MANAGEMENT
# ============================================================================

@cli.group()
def db():
    """Database management commands"""
    pass

@db.command()
@click.option('--force', is_flag=True, help='Force migration without confirmation')
def migrate(force):
    """Run database migrations"""
    if not force:
        click.confirm('Are you sure you want to run database migrations?', abort=True)
    
    async def run_migration():
        console.print("🗄️ Running database migrations...")
        
        migrator = DatabaseMigrator()
        success = migrator.run_migrations()
        
        if success:
            console.print("✅ Database migrations completed successfully")
        else:
            console.print("❌ Database migrations failed")
            sys.exit(1)
    
    asyncio.run(run_migration())

@db.command()
def health():
    """Check database health"""
    async def check_health():
        with console.status("Checking database health..."):
            health_data = await check_database_health()
        
        table = Table(title="Database Health")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        
        for service, info in health_data.items():
            status = info.get("status", "unknown")
            details = json.dumps(info.get("details", {}), indent=2)
            
            table.add_row(service, status, details[:100] + "..." if len(details) > 100 else details)
        
        console.print(table)
    
    asyncio.run(check_health())

@db.command()
@click.option('--backup-file', help='Backup file path')
def backup(backup_file):
    """Backup database"""
    if not backup_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"backup_{timestamp}.sql"
    
    console.print(f"📦 Creating database backup: {backup_file}")
    
    settings = get_settings()
    
    # Create pg_dump command
    cmd = [
        "pg_dump",
        "-h", settings.database.host,
        "-p", str(settings.database.port),
        "-U", settings.database.username,
        "-d", settings.database.database,
        "-f", backup_file,
        "--verbose"
    ]
    
    import subprocess
    try:
        subprocess.run(cmd, check=True, env={"PGPASSWORD": settings.database.password})
        console.print(f"✅ Database backup created: {backup_file}")
    except subprocess.CalledProcessError as e:
        console.print(f"❌ Backup failed: {e}")
        sys.exit(1)

@db.command()
@click.argument('backup_file')
@click.option('--force', is_flag=True, help='Force restore without confirmation')
def restore(backup_file, force):
    """Restore database from backup"""
    if not Path(backup_file).exists():
        console.print(f"❌ Backup file not found: {backup_file}")
        sys.exit(1)
    
    if not force:
        click.confirm(f'This will restore the database from {backup_file}. Continue?', abort=True)
    
    console.print(f"🔄 Restoring database from: {backup_file}")
    
    settings = get_settings()
    
    # Create psql command
    cmd = [
        "psql",
        "-h", settings.database.host,
        "-p", str(settings.database.port),
        "-U", settings.database.username,
        "-d", settings.database.database,
        "-f", backup_file,
        "--verbose"
    ]
    
    import subprocess
    try:
        subprocess.run(cmd, check=True, env={"PGPASSWORD": settings.database.password})
        console.print("✅ Database restored successfully")
    except subprocess.CalledProcessError as e:
        console.print(f"❌ Restore failed: {e}")
        sys.exit(1)

# ============================================================================
# USER MANAGEMENT
# ============================================================================

@cli.group()
def users():
    """User management commands"""
    pass

@users.command()
@click.option('--username', prompt=True, help='Admin username')
@click.option('--password', prompt=True, hide_input=True, help='Admin password')
@click.option('--email', prompt=True, help='Admin email')
def create_admin(username, password, email):
    """Create initial admin user"""
    async def create_admin_user():
        console.print("👤 Creating admin user...")
        
        try:
            await create_initial_admin(username, password, email)
            console.print(f"✅ Admin user created: {username}")
        except Exception as e:
            console.print(f"❌ Failed to create admin user: {e}")
            sys.exit(1)
    
    asyncio.run(create_admin_user())

@users.command()
def list_users():
    """List all users"""
    async def list_all_users():
        console.print("👥 Listing all users...")
        
        # This would require implementing user listing in the backend
        # For now, show a placeholder
        console.print("User listing not implemented yet")
    
    asyncio.run(list_all_users())

# ============================================================================
# COMPONENT MANAGEMENT
# ============================================================================

@cli.group()
def components():
    """Component management commands"""
    pass

@components.command()
def list_components():
    """List available components"""
    async def list_all_components():
        registry = await get_component_registry()
        components = registry.get_all_components()
        
        table = Table(title="Available Components")
        table.add_column("Type", style="cyan")
        table.add_column("Category", style="green")
        table.add_column("Display Name", style="yellow")
        table.add_column("Description", style="white")
        
        for comp_type, metadata in components.items():
            table.add_row(
                comp_type,
                metadata.category.value,
                metadata.display_name,
                metadata.description[:50] + "..." if len(metadata.description) > 50 else metadata.description
            )
        
        console.print(table)
        console.print(f"\nTotal components: {len(components)}")
    
    asyncio.run(list_all_components())

@components.command()
@click.argument('component_type')
def info(component_type):
    """Get detailed component information"""
    async def get_component_info():
        registry = await get_component_registry()
        metadata = registry.get_component_metadata(component_type)
        
        if not metadata:
            console.print(f"❌ Component type not found: {component_type}")
            return
        
        # Create info panel
        info_dict = {
            "type": metadata.component_type,
            "category": metadata.category.value,
            "display_name": metadata.display_name,
            "description": metadata.description,
            "version": metadata.version,
            "author": metadata.author,
            "input_ports": [{"name": p.name, "type": p.port_type.value, "required": p.required} for p in metadata.input_ports],
            "output_ports": [{"name": p.name, "type": p.port_type.value} for p in metadata.output_ports],
            "default_properties": metadata.default_properties
        }
        
        console.print(Panel(JSON.from_data(info_dict), title=f"Component: {component_type}"))
    
    asyncio.run(get_component_info())

# ============================================================================
# AI MANAGEMENT
# ============================================================================

@cli.group()
def ai():
    """AI services management"""
    pass

@ai.command()
def usage():
    """Show AI usage statistics"""
    async def show_usage():
        console.print("🤖 AI Usage Statistics")
        
        try:
            stats = await get_usage_statistics()
            
            table = Table(title="AI Usage Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Requests", str(stats["total_requests"]))
            table.add_row("Total Tokens", f"{stats['total_tokens']:,}")
            table.add_row("Estimated Cost", f"${stats['estimated_total_cost']:.6f}")
            
            console.print(table)
            
            # Show usage by model
            if stats["usage_by_model"]:
                model_table = Table(title="Usage by Model")
                model_table.add_column("Model", style="cyan")
                model_table.add_column("Requests", style="green")
                model_table.add_column("Tokens", style="yellow")
                
                for model, usage in stats["usage_by_model"].items():
                    model_table.add_row(
                        model,
                        str(usage["total_requests"]),
                        f"{usage['total_tokens']:,}"
                    )
                
                console.print(model_table)
        
        except Exception as e:
            console.print(f"❌ Failed to get AI usage statistics: {e}")
    
    asyncio.run(show_usage())

@ai.command()
@click.option('--force', is_flag=True, help='Force reset without confirmation')
def reset_usage(force):
    """Reset AI usage statistics"""
    if not force:
        click.confirm('This will reset all AI usage statistics. Continue?', abort=True)
    
    async def reset_stats():
        console.print("🔄 Resetting AI usage statistics...")
        
        try:
            await reset_usage_statistics()
            console.print("✅ AI usage statistics reset")
        except Exception as e:
            console.print(f"❌ Failed to reset statistics: {e}")
    
    asyncio.run(reset_stats())

# ============================================================================
# MONITORING
# ============================================================================

@cli.group()
def monitor():
    """System monitoring commands"""
    pass

@monitor.command()
@click.option('--hours', default=1, help='Hours of data to show')
def performance(hours):
    """Show performance summary"""
    async def show_performance():
        console.print(f"📊 Performance Summary (last {hours} hour{'s' if hours != 1 else ''})")
        
        try:
            monitor = await get_performance_monitor()
            summary = await monitor.get_performance_summary(hours=hours)
            
            if "error" in summary:
                console.print(f"❌ {summary['error']}")
                return
            
            table = Table(title="Performance Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Current", style="green")
            table.add_column("Average", style="yellow")
            table.add_column("Max", style="red")
            
            for metric_name, data in summary.items():
                if isinstance(data, dict) and "current" in data:
                    table.add_row(
                        metric_name.replace("_", " ").title(),
                        f"{data['current']:.2f}",
                        f"{data['average']:.2f}",
                        f"{data['max']:.2f}"
                    )
            
            console.print(table)
            
            # Show alerts
            alerts = await monitor.get_alerts(resolved=False)
            if alerts:
                console.print(f"\n⚠️  Active Alerts: {len(alerts)}")
                for alert in alerts[:5]:  # Show first 5 alerts
                    level_emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}
                    emoji = level_emoji.get(alert["level"], "❓")
                    console.print(f"  {emoji} {alert['message']}")
            
        except Exception as e:
            console.print(f"❌ Failed to get performance data: {e}")
    
    asyncio.run(show_performance())

@monitor.command()
def alerts():
    """Show current alerts"""
    async def show_alerts():
        console.print("🚨 Current Alerts")
        
        try:
            monitor = await get_performance_monitor()
            alerts = await monitor.get_alerts(resolved=False)
            
            if not alerts:
                console.print("✅ No active alerts")
                return
            
            table = Table(title="Active Alerts")
            table.add_column("Level", style="red")
            table.add_column("Metric", style="cyan")
            table.add_column("Message", style="white")
            table.add_column("Time", style="yellow")
            
            for alert in alerts:
                table.add_row(
                    alert["level"].upper(),
                    alert["metric_name"],
                    alert["message"],
                    alert["timestamp"][:19]  # Remove microseconds
                )
            
            console.print(table)
        
        except Exception as e:
            console.print(f"❌ Failed to get alerts: {e}")
    
    asyncio.run(show_alerts())

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

@cli.group()
def config():
    """Configuration management"""
    pass

@config.command()
def show():
    """Show current configuration"""
    settings = get_settings()
    
    # Mask sensitive data
    config_dict = settings.dict()
    sensitive_keys = ['password', 'secret_key', 'api_key']
    
    def mask_sensitive(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    obj[key] = "***MASKED***" if value else None
                elif isinstance(value, dict):
                    mask_sensitive(value)
    
    mask_sensitive(config_dict)
    
    console.print(Panel(JSON.from_data(config_dict), title="Current Configuration"))

@config.command()
@click.argument('key')
@click.argument('value')
def set_config(key, value):
    """Set configuration value"""
    console.print(f"Setting {key} = {value}")
    console.print("⚠️  Configuration setting not implemented yet")

@config.command()
def validate():
    """Validate current configuration"""
    from backend.config.settings import validate_configuration
    
    settings = get_settings()
    issues = validate_configuration(settings)
    
    if not issues:
        console.print("✅ Configuration is valid")
    else:
        console.print("❌ Configuration issues found:")
        for section, section_issues in issues.items():
            console.print(f"\n[bold red]{section}:[/bold red]")
            for issue in section_issues:
                console.print(f"  • {issue}")

# ============================================================================
# UTILITY COMMANDS
# ============================================================================

@cli.command()
def version():
    """Show version information"""
    settings = get_settings()
    
    info = {
        "application": settings.app_name,
        "version": settings.version,
        "environment": settings.environment.value,
        "debug": settings.debug,
        "python_version": sys.version,
        "features": {
            "ai": settings.enable_ai,
            "websockets": settings.enable_websockets,
            "authentication": settings.enable_authentication
        }
    }
    
    console.print(Panel(JSON.from_data(info), title="Version Information"))

@cli.command()
@click.option('--format', 'output_format', default='json', type=click.Choice(['json', 'yaml']))
@click.option('--output', help='Output file path')
def export_config(output_format, output):
    """Export configuration to file"""
    settings = get_settings()
    config_dict = settings.dict()
    
    if output_format == 'json':
        content = json.dumps(config_dict, indent=2, default=str)
    else:  # yaml
        content = yaml.dump(config_dict, default_flow_style=False)
    
    if output:
        with open(output, 'w') as f:
            f.write(content)
        console.print(f"✅ Configuration exported to: {output}")
    else:
        console.print(content)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n❌ Error: {e}")
        if '--verbose' in sys.argv or '-v' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)

# ============================================================================
# REQUIREMENTS FOR CLI
# ============================================================================

# Add to requirements.txt:
# click>=8.1.0
# rich>=13.0.0
# pyyaml>=6.0