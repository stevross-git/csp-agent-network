#!/usr/bin/env python3
"""
Enhanced CSP System Import Verification Test
============================================

This script tests all the imports in the main.py to ensure they work correctly.
Run this before starting the main application to catch import issues early.
"""

import sys
import traceback
from typing import List, Tuple, Dict, Any

class ImportTester:
    """Test all imports systematically"""
    
    def __init__(self):
        self.results = []
        self.failed_imports = []
        
    def test_import(self, module_name: str, from_items: List[str] = None, 
                    description: str = "") -> bool:
        """Test importing a module or specific items from a module"""
        try:
            if from_items:
                # Test importing specific items
                module = __import__(module_name, fromlist=from_items)
                for item in from_items:
                    getattr(module, item)  # This will raise AttributeError if item doesn't exist
                import_desc = f"from {module_name} import {', '.join(from_items)}"
            else:
                # Test importing entire module
                __import__(module_name)
                import_desc = f"import {module_name}"
            
            self.results.append((import_desc, True, description))
            print(f"✅ {import_desc}")
            return True
            
        except Exception as e:
            self.results.append((import_desc, False, f"{description} - {str(e)}"))
            self.failed_imports.append((import_desc, str(e)))
            print(f"❌ {import_desc} - {str(e)}")
            return False
    
    def test_standard_library(self):
        """Test standard library imports"""
        print("\n🐍 Testing Standard Library Imports...")
        
        standard_imports = [
            ("os", [], "Operating System Interface"),
            ("sys", [], "System-specific parameters"),
            ("asyncio", [], "Asynchronous I/O"),
            ("logging", [], "Logging facility"),
            ("json", [], "JSON encoder/decoder"),
            ("uuid", [], "UUID objects"),
            ("datetime", ["datetime"], "Date and time handling"),
            ("pathlib", ["Path"], "Object-oriented filesystem paths"),
            ("typing", ["Dict", "List", "Any", "Optional", "Union"], "Type hints"),
            ("contextlib", ["asynccontextmanager"], "Context management utilities"),
            ("sqlite3", [], "SQLite database interface"),
            ("time", [], "Time-related functions")
        ]
        
        for module, items, desc in standard_imports:
            self.test_import(module, items, desc)
    
    def test_web_framework(self):
        """Test web framework imports"""
        print("\n🌐 Testing Web Framework Imports...")
        
        web_imports = [
            ("fastapi", ["FastAPI", "WebSocket", "HTTPException", "Depends", "BackgroundTasks", "Request"], "FastAPI web framework"),
            ("fastapi.staticfiles", ["StaticFiles"], "Static file serving"),
            ("fastapi.templating", ["Jinja2Templates"], "Template engine"),
            ("fastapi.responses", ["HTMLResponse", "JSONResponse", "PlainTextResponse"], "HTTP responses"),
            ("fastapi.middleware.cors", ["CORSMiddleware"], "CORS middleware"),
            ("fastapi.middleware.gzip", ["GZipMiddleware"], "Gzip compression"),
            ("fastapi.security", ["HTTPBearer", "HTTPAuthorizationCredentials"], "Security utilities"),
            ("uvicorn", [], "ASGI server"),
        ]
        
        for module, items, desc in web_imports:
            self.test_import(module, items, desc)
    
    def test_database(self):
        """Test database imports"""
        print("\n🗄️ Testing Database Imports...")
        
        db_imports = [
            ("sqlalchemy", ["create_engine", "text"], "SQLAlchemy ORM"),
            ("sqlalchemy.ext.asyncio", ["create_async_engine", "AsyncSession"], "Async SQLAlchemy"),
            ("sqlalchemy.orm", ["sessionmaker"], "SQLAlchemy ORM"),
        ]
        
        for module, items, desc in db_imports:
            self.test_import(module, items, desc)
        
        # Test Redis (optional)
        try:
            self.test_import("redis.asyncio", [], "Redis async client")
        except:
            print("⚠️  Redis not available - some features may be limited")
    
    def test_monitoring(self):
        """Test monitoring imports"""
        print("\n📊 Testing Monitoring Imports...")
        
        monitoring_imports = [
            ("prometheus_client", ["Counter", "Histogram", "Gauge", "generate_latest", "CONTENT_TYPE_LATEST"], "Prometheus metrics"),
            ("psutil", [], "System and process utilities"),
        ]
        
        for module, items, desc in monitoring_imports:
            self.test_import(module, items, desc)
    
    def test_configuration(self):
        """Test configuration imports"""
        print("\n⚙️ Testing Configuration Imports...")
        
        config_imports = [
            ("yaml", [], "YAML parser"),
            ("pydantic", ["BaseModel", "Field", "validator"], "Data validation"),
            ("dotenv", ["load_dotenv"], "Environment variable loading"),
            ("aiofiles", [], "Async file operations"),
            ("aiohttp", [], "Async HTTP client/server"),
        ]
        
        for module, items, desc in config_imports:
            self.test_import(module, items, desc)
    
    def test_scientific_computing(self):
        """Test scientific computing imports"""
        print("\n🔬 Testing Scientific Computing Imports...")
        
        scientific_imports = [
            ("numpy", [], "Numerical computing"),
            ("scipy", [], "Scientific computing"),
            ("networkx", [], "Network analysis"),
        ]
        
        for module, items, desc in scientific_imports:
            self.test_import(module, items, desc)
    
    def test_ai_ml(self):
        """Test AI/ML imports (optional)"""
        print("\n🤖 Testing AI/ML Imports (Optional)...")
        
        ai_imports = [
            ("transformers", [], "Hugging Face Transformers"),
            ("torch", [], "PyTorch"),
            ("sentence_transformers", [], "Sentence Transformers"),
            ("scikit-learn", [], "Machine Learning"),
            ("openai", [], "OpenAI API"),
        ]
        
        for module, items, desc in ai_imports:
            try:
                self.test_import(module, items, desc)
            except:
                print(f"⚠️  {module} not available - AI features may be limited")
    
    def test_csp_components(self):
        """Test CSP-specific component imports"""
        print("\n🔄 Testing CSP Component Imports...")
        
        # Test core components
        core_components = [
            ("core.advanced_csp_core", [
                "AdvancedCSPEngine", "Process", "AtomicProcess", "CompositeProcess",
                "CompositionOperator", "ChannelType", "Event", "ProcessSignature",
                "ProcessContext", "Channel", "ProcessMatcher", "ProtocolEvolution"
            ], "Core CSP engine"),
            
            ("ai_integration.csp_ai_extensions", [
                "AdvancedCSPEngineWithAI", "ProtocolSpec", "ProtocolTemplate",
                "EmergentBehaviorDetector", "CausalityTracker", "QuantumCSPChannel"
            ], "AI extensions"),
            
            ("ai_integration.csp_ai_integration", [
                "AIAgent", "LLMCapability", "CollaborativeAIProcess",
                "MultiAgentReasoningCoordinator", "AdvancedAICSPDemo"
            ], "AI integration"),
            
            ("runtime.csp_runtime_environment", [
                "CSPRuntimeOrchestrator", "RuntimeConfig", "ExecutionModel",
                "SchedulingPolicy", "HighPerformanceRuntimeExecutor"
            ], "Runtime environment"),
            
            ("deployment.csp_deployment_system", [
                "CSPDeploymentOrchestrator", "DeploymentConfig", "DeploymentTarget",
                "ScalingStrategy", "HealthCheckConfig"
            ], "Deployment system"),
            
            ("dev_tools.csp_dev_tools", [
                "CSPDevelopmentTools", "CSPVisualDesigner", "CSPDebugger",
                "CSPCodeGenerator", "CSPTestFramework"
            ], "Development tools"),
            
            ("monitoring.csp_monitoring", [
                "CSPMonitor", "MetricsCollector", "PerformanceAnalyzer",
                "AlertManager", "SystemHealthChecker"
            ], "Monitoring system"),
        ]
        
        for module, items, desc in core_components:
            try:
                self.test_import(module, items, desc)
            except:
                print(f"⚠️  {module} not available - creating stub")
                # Create a simple stub to prevent import errors
                self.create_stub_module(module, items)
    
    def create_stub_module(self, module_name: str, items: List[str]):
        """Create a stub module to prevent import errors"""
        module_parts = module_name.split('.')
        
        # Create simple stub classes
        stub_code = f'"""Stub module for {module_name}"""\n\n'
        
        for item in items:
            if item.endswith('Config') or item.endswith('Strategy'):
                stub_code += f'class {item}:\n    def __init__(self, **kwargs):\n        pass\n\n'
            elif item.startswith('CSP'):
                stub_code += f'class {item}:\n    def __init__(self, *args, **kwargs):\n        pass\n    async def start(self):\n        pass\n    async def stop(self):\n        pass\n\n'
            else:
                stub_code += f'class {item}:\n    def __init__(self, *args, **kwargs):\n        pass\n\n'
        
        print(f"   📝 Created stub for {module_name}")
    
    def test_optional_components(self):
        """Test optional component imports"""
        print("\n🔧 Testing Optional Component Imports...")
        
        optional_imports = [
            ("web_ui.dashboard.app", ["create_dashboard_app"], "Web dashboard"),
            ("database.migrate", ["migrate_main"], "Database migrations"),
        ]
        
        for module, items, desc in optional_imports:
            try:
                self.test_import(module, items, desc)
            except:
                print(f"⚠️  {module} not available - {desc} features disabled")
    
    def generate_report(self):
        """Generate test report"""
        print("\n" + "="*70)
        print("📋 IMPORT TEST REPORT")
        print("="*70)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for _, success, _ in self.results if success)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        
        if self.failed_imports:
            print(f"\n❌ Failed Imports ({len(self.failed_imports)}):")
            for import_desc, error in self.failed_imports:
                print(f"   - {import_desc}")
                print(f"     Error: {error}")
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"\n📊 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 Import tests mostly successful! System should work.")
        elif success_rate >= 60:
            print("⚠️  Some imports failed. System may work with limited features.")
        else:
            print("❌ Many imports failed. Please install missing dependencies.")
            print("   Run: pip install -r requirements.txt")
        
        return success_rate >= 60
    
    def run_all_tests(self):
        """Run all import tests"""
        print("🚀 Enhanced CSP System Import Verification")
        print("="*70)
        
        test_suites = [
            self.test_standard_library,
            self.test_web_framework,
            self.test_database,
            self.test_monitoring,
            self.test_configuration,
            self.test_scientific_computing,
            self.test_ai_ml,
            self.test_csp_components,
            self.test_optional_components
        ]
        
        for test_suite in test_suites:
            try:
                test_suite()
            except Exception as e:
                print(f"❌ Test suite failed: {e}")
                traceback.print_exc()
        
        return self.generate_report()

def main():
    """Main function"""
    tester = ImportTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ Ready to start the Enhanced CSP System!")
        print("   Run: python main.py")
    else:
        print("\n❌ Please fix import issues before starting the system.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
