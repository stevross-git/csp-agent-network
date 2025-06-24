#!/usr/bin/env python3
"""
Enhanced CSP System - Page Directory Scanner
Scans the pages directory and generates a JavaScript file with available pages
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime

def scan_pages_directory(pages_dir):
    """Scan the pages directory for HTML files"""
    pages_dir = Path(pages_dir)
    
    if not pages_dir.exists():
        print(f"❌ Pages directory not found: {pages_dir}")
        return []
    
    html_files = list(pages_dir.glob("*.html"))
    pages = []
    
    for file in html_files:
        page_info = analyze_html_file(file)
        pages.append(page_info)
    
    return sorted(pages, key=lambda x: x['name'])

def analyze_html_file(file_path):
    """Analyze an HTML file to extract metadata"""
    file_path = Path(file_path)
    page_name = file_path.stem
    
    # Read file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
        content = ""
    
    # Extract title
    title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else format_page_name(page_name)
    
    # Extract description from meta tag
    desc_match = re.search(r'<meta\s+name=["\'](description|desc)["\'][^>]*content=["\'](.*?)["\']', content, re.IGNORECASE)
    description = desc_match.group(2).strip() if desc_match else f"Page: {title}"
    
    # Detect category based on content and filename
    category = detect_category(page_name, content)
    
    # Detect if it has auth wrapper
    has_auth = 'auth-wrapper.js' in content or 'cspAuthReady' in content
    
    # Get file stats
    stat = file_path.stat()
    file_size = stat.st_size
    modified_time = datetime.fromtimestamp(stat.st_mtime)
    
    return {
        'name': page_name,
        'title': clean_title(title),
        'description': description[:100] + ('...' if len(description) > 100 else ''),
        'category': category,
        'filename': file_path.name,
        'size': file_size,
        'modified': modified_time.isoformat(),
        'has_auth': has_auth,
        'status': 'available'
    }

def format_page_name(name):
    """Format page name for display"""
    return name.replace('-', ' ').replace('_', ' ').title()

def clean_title(title):
    """Clean HTML title"""
    # Remove common prefixes/suffixes
    title = re.sub(r'^(Enhanced CSP System\s*[-–—]?\s*)', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s*[-–—]?\s*(Enhanced CSP System)$', '', title, flags=re.IGNORECASE)
    return title.strip()

def detect_category(page_name, content):
    """Detect page category based on name and content"""
    name_lower = page_name.lower()
    content_lower = content.lower()
    
    # Security category
    if any(keyword in name_lower for keyword in ['security', 'auth', 'login', 'audit', 'alert', 'firewall', 'cert']):
        return 'security'
    
    # AI category
    if any(keyword in name_lower for keyword in ['ai', 'neural', 'quantum', 'blockchain', 'agent', 'ml']):
        return 'ai'
    
    # Admin category
    if any(keyword in name_lower for keyword in ['admin', 'user', 'role', 'permission', 'setting']):
        return 'admin'
    
    # Infrastructure category
    if any(keyword in name_lower for keyword in ['deploy', 'infra', 'container', 'kubernetes', 'docker']):
        return 'infrastructure'
    
    # Monitoring category
    if any(keyword in name_lower for keyword in ['monitor', 'metric', 'log', 'analytic', 'report']):
        return 'monitoring'
    
    # Core category (default for main pages)
    if any(keyword in name_lower for keyword in ['dashboard', 'index', 'main', 'home', 'designer']):
        return 'core'
    
    # Check content for category hints
    if any(keyword in content_lower for keyword in ['authentication', 'security', 'audit']):
        return 'security'
    
    if any(keyword in content_lower for keyword in ['artificial intelligence', 'machine learning', 'neural']):
        return 'ai'
    
    return 'other'

def get_icon_for_category(category, page_name):
    """Get icon for page based on category and name"""
    name_lower = page_name.lower()
    
    # Specific page icons
    specific_icons = {
        'dashboard': '📊',
        'admin': '👑',
        'login': '🔑',
        'security': '🔐',
        'monitoring': '📈',
        'designer': '🎨',
        'settings': '⚙️',
        'ai-agents': '🤖',
        'quantum': '⚛️',
        'blockchain': '⛓️',
        'neural': '🧠',
        'api': '🔍',
        'deployment': '🚀',
        'infrastructure': '🏗️',
        'containers': '📦',
        'kubernetes': '☸️',
        'chat': '💬',
        'collaboration': '🤝',
        'notifications': '🔔',
        'audit': '🔍',
        'alerts': '🚨',
        'reports': '📈',
        'analytics': '📊',
        'users': '👥',
        'roles': '🎭',
        'permissions': '🔐',
        'logs': '📄',
        'backup': '💾',
        'recovery': '🔄',
        'integrations': '🔗',
        'webhooks': '🪝',
        'certificates': '📜',
        'firewall': '🛡️',
        'vpn': '🔒'
    }
    
    # Check for specific page name match
    for key, icon in specific_icons.items():
        if key in name_lower:
            return icon
    
    # Category-based icons
    category_icons = {
        'core': '🎯',
        'security': '🔐',
        'ai': '🤖',
        'admin': '👑',
        'infrastructure': '🏗️',
        'monitoring': '📈',
        'other': '📄'
    }
    
    return category_icons.get(category, '📄')

def generate_pages_js(pages, output_file):
    """Generate JavaScript file with page definitions"""
    js_content = """// Auto-generated page definitions
// Generated on: {timestamp}
// Total pages: {total_pages}

const availablePages = {pages_json};

const pageCategories = {categories_json};

const pageIcons = {icons_json};

// Export for use in index.html
if (typeof window !== 'undefined') {{
    window.availablePages = availablePages;
    window.pageCategories = pageCategories;
    window.pageIcons = pageIcons;
    console.log('📄 Loaded {total_pages} available pages');
}}
""".format(
        timestamp=datetime.now().isoformat(),
        total_pages=len(pages),
        pages_json=json.dumps(pages, indent=2),
        categories_json=json.dumps({cat: [p['name'] for p in pages if p['category'] == cat] 
                                  for cat in set(p['category'] for p in pages)}, indent=2),
        icons_json=json.dumps({p['name']: get_icon_for_category(p['category'], p['name']) 
                              for p in pages}, indent=2)
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    print(f"✅ Generated: {output_file}")

def main():
    """Main function"""
    print("🔍 Enhanced CSP System - Page Scanner")
    print("=" * 50)
    
    # Define paths
    current_dir = Path(__file__).parent
    pages_dir = current_dir / "pages"
    output_file = current_dir / "js" / "available-pages.js"
    
    # Create js directory if it doesn't exist
    output_file.parent.mkdir(exist_ok=True)
    
    print(f"📁 Scanning: {pages_dir}")
    print(f"📝 Output: {output_file}")
    print()
    
    # Scan pages
    pages = scan_pages_directory(pages_dir)
    
    if not pages:
        print("❌ No HTML pages found!")
        return
    
    # Display results
    print(f"✅ Found {len(pages)} pages:")
    print()
    
    # Group by category
    categories = {}
    for page in pages:
        cat = page['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(page)
    
    for category, cat_pages in sorted(categories.items()):
        print(f"📂 {category.upper()} ({len(cat_pages)} pages)")
        for page in cat_pages:
            icon = get_icon_for_category(page['category'], page['name'])
            auth_status = "🔐" if page['has_auth'] else "🔓"
            size_kb = page['size'] // 1024
            print(f"   {icon} {auth_status} {page['name']:<20} - {page['title']}")
            print(f"      {page['description']}")
            print(f"      Size: {size_kb}KB, Modified: {page['modified'][:10]}")
            print()
    
    # Generate JavaScript file
    generate_pages_js(pages, output_file)
    
    print(f"🎉 Scan complete! Generated {output_file}")
    print()
    print("💡 Tips:")
    print("   • Include this JS file in your index.html")
    print("   • Re-run this script when you add new pages")
    print("   • Pages with 🔐 have authentication enabled")
    print("   • Pages with 🔓 need authentication wrapper")

if __name__ == "__main__":
    main()