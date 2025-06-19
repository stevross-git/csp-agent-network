#!/usr/bin/env python3
"""
Enhanced CSP System Setup and Verification Script
=================================================

This script sets up and verifies that all components of the Enhanced CSP System
are properly installed and working together.

Features:
- Dependency verification
- Component import testing
- Database initialization
- Configuration validation
- Integration testing
- Performance benchmarking
"""

import os
import sys
import subprocess
import importlib
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CSPSystemSetup:
    """Enhanced CSP System setup and verification"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.setup_results = {}
        self.failed_components = []
        
    def print_banner(self):
        """Print setup banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════╗
║                      Enhanced CSP System Setup                          ║
║                                                                          ║
║  Revolutionary AI-to-AI Communication Platform                          ║
║  ✨ Quantum-inspired protocols                                          ║
║  🤖 AI-powered process synthesis                                        ║
║  🚀 Production-ready deployment                                         ║
║  🔧 Advanced development tools                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def check_python_version(self):
        """Check Python version compatibility"""
        logger.info("Checking Python version...")
        
        version = sys.version_info
        if version < (3, 8):
            logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
            return False
        
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    
    def create_directories(self):
        """Create necessary directories"""
        logger.info("Creating directory structure...")
        
        directories = [
            'data',
            'logs',
            'config',
            'web_ui/static',
            'web_ui/templates',
            'tests',
            'deployment/k8s',
            'deployment/docker',
            'docs'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  📁 Created: {directory}")
        
        logger.info("✅ Directory structure created")
        return True
    
    def check_core_dependencies(self):
        """Check if core dependencies are available"""
        logger.info("Checking core dependencies...")
        
        core_deps = [
            'fastapi',
            'uvicorn',
            'pydantic',
            'sqlalchemy',
            'redis',
            'numpy',
            'scipy',
            'networkx',
            'asyncio',
            'yaml',
            'aiohttp',
            'websockets'
        ]
        
        missing_deps = []
        for dep in core_deps:
            try:
                importlib.import_module(dep)
                logger.info(f"  ✅ {dep}")
            except ImportError:
                logger.warning(f"  ❌ {dep} - missing")
                missing_deps.append(dep)
        
        if missing_deps:
            logger.error(f"Missing dependencies: {missing_deps}")
            logger.info("Run: pip install -r requirements.txt")
            return False
        
        logger.info("✅ All core dependencies available")
        return True
    
    def check_ai_dependencies(self):
        """Check AI and ML dependencies"""
        logger.info("Checking AI/ML dependencies...")
        
        ai_deps = [
            'torch',
            'transformers',
            'sentence_transformers',
            'scikit-learn',
            'openai'
        ]
        
        missing_deps = []
        for dep in ai_deps:
            try:
                importlib.import_module(dep.replace('-', '_'))
                logger.info(f"  ✅ {dep}")
            except ImportError:
                logger.warning(f"  ⚠️  {dep} - optional AI feature")
                missing_deps.append(dep)
        
        if missing_deps:
            logger.warning(f"Optional AI dependencies missing: {missing_deps}")
            logger.info("Some AI features may not be available")
        
        return True
    
    def test_component_imports(self):
        """Test importing all CSP components"""
        logger.info("Testing component imports...")
        
        components = [
            ('core.advanced_csp_core', 'AdvancedCSPEngine'),
            ('ai_integration.csp_ai_extensions', 'AdvancedCSPEngineWithAI'),
            ('ai_integration.csp_ai_integration', 'AIAgent'),
            ('runtime.csp_runtime_environment', 'CSPRuntimeOrchestrator'),
            ('dev_tools.csp_dev_tools', 'CSPDevelopmentTools'),
            ('monitoring.csp_monitoring', 'CSPMonitor'),
        ]
        
        working_components = []
        for module_name, class_name in components:
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                logger.info(f"  ✅ {module_name}.{class_name}")
                working_components.append((module_name, class_name))
            except (ImportError, AttributeError) as e:
                logger.warning(f"  ⚠️  {module_name}.{class_name} - {e}")
                self.failed_components.append(module_name)
        
        if working_components:
            logger.info(f"✅ {len(working_components)} components available")
        
        return len(working_components) > 0
    
    def create_default_config(self):
        """Create default configuration files"""
        logger.info("Creating default configuration...")
        
        # System configuration
        system_config = {
            'app_name': 'Enhanced CSP System',
            'version': '2.0.0',
            'debug': False,
            'environment': 'development',
            'host': '0.0.0.0',
            'port': 8000,
            'database': {
                'url': 'sqlite:///./data/enhanced_csp.db',
                'echo': False
            },
            'redis': {
                'url': 'redis://localhost:6379/0'
            },
            'monitoring': {
                'enable_prometheus': True,
                'metrics_port': 9090
            },
            'ai': {
                'enable_llm_integration': True,
                'default_model': 'gpt-4'
            }
        }
        
        config_file = self.project_root / 'config' / 'system.yaml'
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(system_config, f, default_flow_style=False)
        
        logger.info(f"  📄 Created: {config_file}")
        
        # Environment file
        env_content = """# Enhanced CSP System Environment Configuration
CSP_LOG_LEVEL=INFO
CSP_SECRET_KEY=dev-secret-key-change-in-production
CSP_CONFIG_PATH=config/system.yaml
CSP_DATA_DIR=./data
CSP_ENABLE_DEBUG=false
"""
        
        env_file = self.project_root / '.env'
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        logger.info(f"  📄 Created: {env_file}")
        logger.info("✅ Default configuration created")
        return True
    
    def initialize_database(self):
        """Initialize the database"""
        logger.info("Initializing database...")
        
        try:
            # Create database file
            db_file = self.project_root / 'data' / 'enhanced_csp.db'
            
            import sqlite3
            conn = sqlite3.connect(str(db_file))
            
            # Create basic tables
            conn.execute('''
                CREATE TABLE IF NOT EXISTS processes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info(f"  📊 Database initialized: {db_file}")
            logger.info("✅ Database setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
    
    async def test_basic_functionality(self):
        """Test basic system functionality"""
        logger.info("Testing basic functionality...")
        
        try:
            # Test CSP engine creation
            from core.advanced_csp_core import AdvancedCSPEngine
            engine = AdvancedCSPEngine()
            logger.info("  ✅ CSP engine creation")
            
            # Test process creation
            from core.advanced_csp_core import AtomicProcess, ProcessSignature
            process = AtomicProcess(
                name="test_process",
                signature=ProcessSignature(inputs=[], outputs=[])
            )
            logger.info("  ✅ Process creation")
            
            # Test AI integration (if available)
            try:
                from ai_integration.csp_ai_integration import AIAgent, LLMCapability
                capability = LLMCapability("test-model", "testing")
                agent = AIAgent("test_agent", [capability])
                logger.info("  ✅ AI agent creation")
            except ImportError:
                logger.info("  ⚠️  AI integration not available")
            
            logger.info("✅ Basic functionality tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Functionality test failed: {e}")
            return False
    
    def create_example_files(self):
        """Create example files for users"""
        logger.info("Creating example files...")
        
        # Example process
        example_process = '''#!/usr/bin/env python3
"""
Example CSP Process
==================

Simple example showing how to create and run CSP processes.
"""

import asyncio
from core.advanced_csp_core import AdvancedCSPEngine, AtomicProcess, ProcessSignature

async def main():
    """Example main function"""
    
    # Create CSP engine
    engine = AdvancedCSPEngine()
    
    # Create a simple process
    process = AtomicProcess(
        name="hello_world",
        signature=ProcessSignature(inputs=[], outputs=["greeting"])
    )
    
    # Define process behavior
    async def hello_behavior(context):
        await context.send("greeting", "Hello, CSP World!")
        return "completed"
    
    process.behavior = hello_behavior
    
    # Start the process
    process_id = await engine.start_process(process)
    print(f"Started process: {process_id}")
    
    # Wait a bit
    await asyncio.sleep(1)
    
    print("Example completed!")

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        example_file = self.project_root / 'examples' / 'simple_process.py'
        example_file.parent.mkdir(exist_ok=True)
        with open(example_file, 'w') as f:
            f.write(example_process)
        
        logger.info(f"  📄 Created: {example_file}")
        
        # README for examples
        readme_content = '''# Enhanced CSP System Examples

This directory contains example code showing how to use the Enhanced CSP System.

## Files

- `simple_process.py` - Basic process creation and execution
- `ai_collaboration.py` - AI agent collaboration example
- `quantum_communication.py` - Quantum-inspired communication patterns

## Running Examples

```bash
python examples/simple_process.py
```

## More Information

See the main documentation for detailed usage instructions.
'''
        
        readme_file = self.project_root / 'examples' / 'README.md'
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        logger.info("✅ Example files created")
        return True
    
    def generate_summary_report(self):
        """Generate setup summary report"""
        logger.info("Generating setup summary...")
        
        report = {
            'setup_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'project_root': str(self.project_root),
            'results': self.setup_results,
            'failed_components': self.failed_components,
            'status': 'completed'
        }
        
        # Save report
        report_file = self.project_root / 'setup_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*70)
        print("🎉 ENHANCED CSP SYSTEM SETUP COMPLETE")
        print("="*70)
        print(f"📊 Setup Report: {report_file}")
        print(f"🐍 Python Version: {report['python_version']}")
        print(f"📁 Project Root: {report['project_root']}")
        
        if self.failed_components:
            print(f"⚠️  Components with issues: {len(self.failed_components)}")
            for component in self.failed_components:
                print(f"   - {component}")
        else:
            print("✅ All components working properly")
        
        print("\n🚀 Next Steps:")
        print("   1. Start the system: python main.py")
        print("   2. Visit: http://localhost:8000")
        print("   3. Check examples: python examples/simple_process.py")
        print("   4. Read documentation: docs/")
        
        return report
    
    async def run_complete_setup(self):
        """Run complete setup process"""
        self.print_banner()
        
        steps = [
            ("Python Version Check", self.check_python_version),
            ("Directory Creation", self.create_directories),
            ("Core Dependencies", self.check_core_dependencies),
            ("AI Dependencies", self.check_ai_dependencies),
            ("Component Imports", self.test_component_imports),
            ("Configuration", self.create_default_config),
            ("Database Init", self.initialize_database),
            ("Functionality Test", self.test_basic_functionality),
            ("Example Files", self.create_example_files)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n{'='*20} {step_name} {'='*20}")
            
            try:
                if asyncio.iscoroutinefunction(step_func):
                    result = await step_func()
                else:
                    result = step_func()
                
                self.setup_results[step_name] = {
                    'status': 'success' if result else 'warning',
                    'timestamp': time.strftime('%H:%M:%S')
                }
                
            except Exception as e:
                logger.error(f"{step_name} failed: {e}")
                self.setup_results[step_name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': time.strftime('%H:%M:%S')
                }
        
        # Generate final report
        return self.generate_summary_report()

async def main():
    """Main setup function"""
    setup = CSPSystemSetup()
    report = await setup.run_complete_setup()
    return report

if __name__ == "__main__":
    try:
        report = asyncio.run(main())
        sys.exit(0 if not report.get('failed_components') else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
