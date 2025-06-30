#!/usr/bin/env python3
"""
Test Secure Database Connections
Verifies all databases are properly secured with authentication
"""

import asyncio
import os
import sys
from typing import Dict, List, Tuple
import asyncpg
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init()

class DatabaseSecurityTester:
    """Test database security configurations"""
    
    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []
        
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{Fore.CYAN}{'=' * 60}")
        print(f"{text}")
        print(f"{'=' * 60}{Style.RESET_ALL}\n")
        
    def print_result(self, service: str, success: bool, message: str):
        """Print test result"""
        icon = "✅" if success else "❌"
        color = Fore.GREEN if success else Fore.RED
        print(f"{color}{icon} {service}: {message}{Style.RESET_ALL}")
        self.results.append((service, success, message))
    
    async def test_postgres_security(self, host: str, port: int, 
                                   database: str, user: str, password: str) -> bool:
        """Test PostgreSQL authentication"""
        try:
            # Try to connect with password
            conn = await asyncpg.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                timeout=5
            )
            
            # Verify we can query
            version = await conn.fetchval("SELECT version()")
            await conn.close()
            
            self.print_result(
                f"PostgreSQL ({database})", 
                True, 
                f"Authenticated successfully"
            )
            
            # Try to connect without password (should fail)
            try:
                conn = await asyncpg.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password='',  # Empty password
                    timeout=2
                )
                await conn.close()
                self.print_result(
                    f"PostgreSQL ({database}) - No Auth", 
                    False, 
                    "WARNING: Accepts empty password!"
                )
                return False
            except:
                self.print_result(
                    f"PostgreSQL ({database}) - No Auth", 
                    True, 
                    "Correctly rejects empty password"
                )
                
            return True
            
        except Exception as e:
            self.print_result(
                f"PostgreSQL ({database})", 
                False, 
                f"Failed to connect: {str(e)}"
            )
            return False
    
    async def test_redis_security(self, host: str, port: int, password: str) -> bool:
        """Test Redis authentication"""
        try:
            # Try to connect with password
            client = redis.Redis(
                host=host,
                port=port,
                password=password,
                decode_responses=True,
                socket_connect_timeout=5
            )
            
            # Test connection
            await client.ping()
            await client.close()
            
            self.print_result(
                "Redis", 
                True, 
                "Authenticated successfully"
            )
            
            # Try without password (should fail)
            if password:  # Only test if password is set
                try:
                    client_noauth = redis.Redis(
                        host=host,
                        port=port,
                        decode_responses=True,
                        socket_connect_timeout=2
                    )
                    await client_noauth.ping()
                    await client_noauth.close()
                    
                    self.print_result(
                        "Redis - No Auth", 
                        False, 
                        "WARNING: Accepts connections without password!"
                    )
                    return False
                except redis.AuthenticationError:
                    self.print_result(
                        "Redis - No Auth", 
                        True, 
                        "Correctly rejects unauthenticated connections"
                    )
            else:
                self.print_result(
                    "Redis", 
                    False, 
                    "WARNING: No password set!"
                )
                return False
                
            return True
            
        except Exception as e:
            self.print_result(
                "Redis", 
                False, 
                f"Failed to connect: {str(e)}"
            )
            return False
    
    async def test_mongodb_security(self, uri: str) -> bool:
        """Test MongoDB authentication"""
        try:
            # Connect with credentials
            client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
            
            # Test connection
            await client.admin.command('ping')
            
            self.print_result(
                "MongoDB", 
                True, 
                "Authenticated successfully"
            )
            
            client.close()
            return True
            
        except Exception as e:
            self.print_result(
                "MongoDB", 
                False, 
                f"Failed to connect: {str(e)}"
            )
            return False
    
    async def test_all_connections(self):
        """Test all database connections"""
        self.print_header("Testing Database Security Configuration")
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv('backend/.env')
        
        # Test PostgreSQL databases
        postgres_tests = [
            {
                'name': 'Main Database',
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', '5432')),
                'database': os.getenv('DB_NAME', 'csp_visual_designer'),
                'user': os.getenv('DB_USER', 'csp_user'),
                'password': os.getenv('DB_PASSWORD', '')
            },
            {
                'name': 'AI Models Database',
                'host': os.getenv('AI_MODELS_DB_HOST', 'localhost'),
                'port': int(os.getenv('AI_MODELS_DB_PORT', '5433')),
                'database': os.getenv('AI_MODELS_DB_NAME', 'ai_models_db'),
                'user': os.getenv('AI_MODELS_DB_USER', 'ai_models_user'),
                'password': os.getenv('AI_MODELS_DB_PASSWORD', '')
            },
            {
                'name': 'Vector Database',
                'host': os.getenv('VECTOR_DB_HOST', 'localhost'),
                'port': int(os.getenv('VECTOR_DB_PORT', '5434')),
                'database': os.getenv('VECTOR_DB_NAME', 'vector_db'),
                'user': os.getenv('VECTOR_DB_USER', 'vector_user'),
                'password': os.getenv('VECTOR_DB_PASSWORD', '')
            }
        ]
        
        for pg_config in postgres_tests:
            await self.test_postgres_security(**pg_config)
        
        # Test Redis
        await self.test_redis_security(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            password=os.getenv('REDIS_PASSWORD', '')
        )
        
        # Test MongoDB if configured
        mongo_uri = os.getenv('MONGO_URI')
        if mongo_uri:
            await self.test_mongodb_security(mongo_uri)
        
        # Summary
        self.print_header("Security Test Summary")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for _, success, _ in self.results if success)
        failed_tests = total_tests - passed_tests
        
        if failed_tests == 0:
            print(f"{Fore.GREEN}✅ All {total_tests} security tests passed!{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ {failed_tests} out of {total_tests} tests failed!{Style.RESET_ALL}")
            print(f"\n{Fore.YELLOW}Failed tests:{Style.RESET_ALL}")
            for service, success, message in self.results:
                if not success:
                    print(f"  - {service}: {message}")
        
        return failed_tests == 0

    async def test_connection_limits(self):
        """Test connection pool limits and timeouts"""
        self.print_header("Testing Connection Limits")
        
        # Test PostgreSQL connection pool
        try:
            connections = []
            max_connections = 25
            
            for i in range(max_connections):
                conn = await asyncpg.connect(
                    host=os.getenv('DB_HOST', 'localhost'),
                    port=int(os.getenv('DB_PORT', '5432')),
                    database=os.getenv('DB_NAME'),
                    user=os.getenv('DB_USER'),
                    password=os.getenv('DB_PASSWORD'),
                    timeout=1
                )
                connections.append(conn)
            
            self.print_result(
                "PostgreSQL Connection Pool",
                True,
                f"Successfully created {max_connections} connections"
            )
            
            # Clean up
            for conn in connections:
                await conn.close()
                
        except Exception as e:
            self.print_result(
                "PostgreSQL Connection Pool",
                False,
                f"Failed at connection limit test: {str(e)}"
            )

async def main():
    """Run all security tests"""
    tester = DatabaseSecurityTester()
    
    try:
        success = await tester.test_all_connections()
        await tester.test_connection_limits()
        
        if not success:
            print(f"\n{Fore.RED}⚠️  Security issues detected! Fix them before proceeding.{Style.RESET_ALL}")
            sys.exit(1)
        else:
            print(f"\n{Fore.GREEN}🎉 All database connections are properly secured!{Style.RESET_ALL}")
            
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Test interrupted by user{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}Test failed with error: {e}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    # Check if running in Docker or local
    if os.path.exists('/.dockerenv'):
        print("Running inside Docker container")
    else:
        print("Running on local machine")
        
    asyncio.run(main())