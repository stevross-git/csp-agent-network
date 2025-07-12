#!/bin/bash
# enhanced_csp/network/scripts/install_all_optimizations.sh
"""
Complete installation script for all CSP network optimizations.
Installs dependencies, configures system, and sets up hardware acceleration.
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root for security reasons."
        log_info "Some commands will use sudo when needed."
        exit 1
    fi
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            OS="debian"
            log_info "Detected Debian/Ubuntu system"
        elif [ -f /etc/redhat-release ]; then
            OS="redhat"
            log_info "Detected Red Hat/CentOS/Fedora system"
        else
            OS="linux"
            log_info "Detected generic Linux system"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        log_info "Detected macOS system"
    else
        OS="unknown"
        log_warning "Unknown operating system: $OSTYPE"
    fi
}

# Install basic dependencies
install_basic_dependencies() {
    log_info "Installing basic Python dependencies..."
    
    # Upgrade pip first
    python3 -m pip install --upgrade pip
    
    # Core dependencies
    local core_deps=(
        "numpy>=1.21.0"
        "numba>=0.56.0"
        "psutil>=5.8.0"
        "asyncio-mqtt>=0.11.0"
        "aiofiles>=0.8.0"
    )
    
    python3 -m pip install "${core_deps[@]}"
    log_success "Basic dependencies installed"
}

# Install performance dependencies
install_performance_dependencies() {
    log_info "Installing performance optimization dependencies..."
    
    local perf_deps=(
        "orjson>=3.8.0"          # Fast JSON
        "msgpack>=1.0.4"         # Binary serialization
        "lz4>=4.0.0"             # Fast compression
        "zstandard>=0.18.0"      # High-ratio compression
        "brotli>=1.0.9"          # Web compression
        "uvloop>=0.17.0"         # Fast event loop
        "aioquic>=0.9.20"        # QUIC protocol
        "cryptography>=37.0.0"   # Cryptographic functions
        "pyzmq>=23.0.0"          # ZeroMQ messaging
    )
    
    for dep in "${perf_deps[@]}"; do
        log_info "Installing $dep..."
        python3 -m pip install "$dep" || log_warning "Failed to install $dep (optional)"
    done
    
    log_success "Performance dependencies installed"
}

# Install GPU acceleration (CUDA)
install_cuda_support() {
    log_info "Installing CUDA support for GPU acceleration..."
    
    # Check if NVIDIA GPU is present
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU detected, installing CUDA support..."
        
        # Install CuPy (CUDA Python)
        python3 -m pip install cupy-cuda11x || {
            log_warning "CuPy installation failed, trying alternative..."
            python3 -m pip install cupy-cuda12x || {
                log_warning "CUDA support installation failed (optional)"
                return 1
            }
        }
        
        # Install additional CUDA libraries
        python3 -m pip install numba[cuda] || log_warning "Numba CUDA support failed (optional)"
        
        log_success "CUDA support installed"
        return 0
    else
        log_warning "No NVIDIA GPU detected, skipping CUDA installation"
        return 1
    fi
}

# Install DPDK dependencies
install_dpdk_dependencies() {
    log_info "Installing DPDK dependencies..."
    
    case $OS in
        "debian")
            sudo apt-get update
            sudo apt-get install -y \
                dpdk dpdk-dev \
                libnuma-dev \
                libpcap-dev \
                pkg-config \
                build-essential
            ;;
        "redhat")
            sudo yum install -y \
                dpdk dpdk-devel \
                numactl-devel \
                libpcap-devel \
                pkgconfig \
                gcc gcc-c++ make
            ;;
        *)
            log_warning "DPDK installation not supported on this OS"
            return 1
            ;;
    esac
    
    # Install Python DPDK bindings (if available)
    python3 -m pip install dpdk-python || log_warning "Python DPDK bindings not available"
    
    log_success "DPDK dependencies installed"
}

# Install RDMA dependencies
install_rdma_dependencies() {
    log_info "Installing RDMA dependencies..."
    
    case $OS in
        "debian")
            sudo apt-get install -y \
                libibverbs-dev \
                librdmacm-dev \
                rdma-core \
                infiniband-diags
            ;;
        "redhat")
            sudo yum install -y \
                libibverbs-devel \
                librdmacm-devel \
                rdma-core \
                infiniband-diags
            ;;
        *)
            log_warning "RDMA installation not supported on this OS"
            return 1
            ;;
    esac
    
    # Install Python RDMA bindings
    python3 -m pip install pyverbs || log_warning "PyVerbs installation failed (optional)"
    
    log_success "RDMA dependencies installed"
}

# Configure system for performance
configure_system_performance() {
    log_info "Configuring system for maximum network performance..."
    
    # Create backup of original sysctl.conf
    sudo cp /etc/sysctl.conf /etc/sysctl.conf.backup.$(date +%Y%m%d_%H%M%S)
    
    # Network performance tuning
    cat << EOF | sudo tee -a /etc/sysctl.conf
# CSP Network Performance Optimizations
# Added by install_all_optimizations.sh

# Increase network buffer sizes
net.core.rmem_default = 262144
net.core.rmem_max = 134217728
net.core.wmem_default = 262144
net.core.wmem_max = 134217728

# TCP buffer sizes
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728

# Increase maximum socket listen() backlog
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 30000

# Increase number of incoming connections
net.ipv4.tcp_max_syn_backlog = 65535

# Enable TCP window scaling
net.ipv4.tcp_window_scaling = 1

# Enable selective acknowledgments
net.ipv4.tcp_sack = 1

# Increase local port range
net.ipv4.ip_local_port_range = 1024 65535

# Reduce TIME_WAIT timeout
net.ipv4.tcp_fin_timeout = 30

# Enable TCP fast open
net.ipv4.tcp_fastopen = 3

# Disable TCP slow start restart
net.ipv4.tcp_slow_start_after_idle = 0

# Increase maximum number of open files
fs.file-max = 2097152

EOF
    
    # Apply changes immediately
    sudo sysctl -p
    
    log_success "System performance configuration applied"
}

# Setup huge pages for DPDK
setup_huge_pages() {
    log_info "Setting up huge pages for DPDK..."
    
    # Configure 2MB huge pages
    echo 1024 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
    
    # Mount huge page filesystem
    sudo mkdir -p /mnt/huge
    sudo mount -t hugetlbfs nodev /mnt/huge
    
    # Make it persistent
    echo "nodev /mnt/huge hugetlbfs defaults 0 0" | sudo tee -a /etc/fstab
    
    log_success "Huge pages configured"
}

# Configure CPU isolation for network processing
configure_cpu_isolation() {
    log_info "Configuring CPU core isolation for network processing..."
    
    local cpu_count=$(nproc)
    local network_cores=""
    
    if [ $cpu_count -ge 8 ]; then
        # Isolate last 2 cores for network processing
        local start_core=$((cpu_count - 2))
        local end_core=$((cpu_count - 1))
        network_cores="${start_core},${end_core}"
        
        log_info "Isolating CPU cores $network_cores for network processing"
        
        # Update GRUB configuration
        local grub_line="isolcpus=${network_cores} nohz_full=${network_cores} rcu_nocbs=${network_cores}"
        
        sudo sed -i "s/GRUB_CMDLINE_LINUX=\"/GRUB_CMDLINE_LINUX=\"${grub_line} /" /etc/default/grub
        sudo update-grub || sudo grub2-mkconfig -o /boot/grub2/grub.cfg
        
        log_warning "CPU isolation configured. Reboot required to take effect."
    else
        log_warning "Not enough CPU cores for isolation (need at least 8, have $cpu_count)"
    fi
}

# Setup network interface optimizations
setup_network_interface_optimizations() {
    log_info "Setting up network interface optimizations..."
    
    # Get the primary network interface
    local primary_iface=$(ip route | grep default | awk '{print $5}' | head -n1)
    
    if [ -n "$primary_iface" ]; then
        log_info "Optimizing network interface: $primary_iface"
        
        # Increase ring buffer sizes
        sudo ethtool -G $primary_iface rx 4096 tx 4096 2>/dev/null || log_warning "Could not set ring buffer sizes"
        
        # Enable checksum offloading
        sudo ethtool -K $primary_iface rx on tx on 2>/dev/null || log_warning "Could not enable checksum offloading"
        
        # Enable TCP segmentation offload
        sudo ethtool -K $primary_iface tso on 2>/dev/null || log_warning "Could not enable TSO"
        
        # Enable generic receive offload
        sudo ethtool -K $primary_iface gro on 2>/dev/null || log_warning "Could not enable GRO"
        
        log_success "Network interface optimizations applied"
    else
        log_warning "Could not detect primary network interface"
    fi
}

# Create performance monitoring script
create_performance_monitor() {
    log_info "Creating performance monitoring script..."
    
    cat << 'EOF' > performance_monitor.sh
#!/bin/bash
# CSP Network Performance Monitor

echo "=== CSP Network Performance Monitor ==="
echo "Timestamp: $(date)"
echo

echo "=== CPU Usage ==="
top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"% CPU Usage"}'

echo
echo "=== Memory Usage ==="
free -h

echo
echo "=== Network Statistics ==="
ss -tuln | head -10

echo
echo "=== Network Interface Statistics ==="
cat /proc/net/dev | grep -v "Inter-" | grep -v "face"

echo
echo "=== Huge Pages Status ==="
cat /proc/meminfo | grep -i huge

echo
echo "=== DPDK Status ==="
if command -v dpdk-devbind.py &> /dev/null; then
    dpdk-devbind.py --status
else
    echo "DPDK not installed"
fi

echo
echo "=== GPU Status ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
else
    echo "NVIDIA GPU not available"
fi
EOF
    
    chmod +x performance_monitor.sh
    log_success "Performance monitor created: ./performance_monitor.sh"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    echo
    echo "=== Python Dependencies ==="
    python3 -c "
import sys
deps = [
    ('numpy', 'NumPy'),
    ('numba', 'Numba'),
    ('orjson', 'orjson'),
    ('msgpack', 'MessagePack'),
    ('lz4', 'LZ4'),
    ('zstandard', 'Zstandard'),
    ('uvloop', 'uvloop'),
    ('aioquic', 'aioquic'),
]

for module, name in deps:
    try:
        __import__(module)
        print(f'✅ {name}')
    except ImportError:
        print(f'❌ {name}')
"
    
    echo
    echo "=== GPU Support ==="
    python3 -c "
try:
    import cupy
    print('✅ CuPy (CUDA)')
except ImportError:
    print('❌ CuPy (CUDA)')

try:
    import pyopencl
    print('✅ PyOpenCL')
except ImportError:
    print('❌ PyOpenCL')
"
    
    echo
    echo "=== System Configuration ==="
    echo "Network buffers: $(sysctl net.core.rmem_max | awk '{print $3}')"
    echo "Huge pages: $(cat /proc/meminfo | grep HugePages_Total | awk '{print $2}')"
    echo "CPU cores: $(nproc)"
    
    log_success "Installation verification complete"
}

# Main installation flow
main() {
    echo "======================================"
    echo "CSP Network Optimization Installer"
    echo "======================================"
    echo
    
    check_root
    detect_os
    
    log_info "Starting installation process..."
    
    # Basic installations (always run)
    install_basic_dependencies
    install_performance_dependencies
    
    # Optional installations (may fail gracefully)
    install_cuda_support
    
    # System-level optimizations (require sudo)
    log_info "System-level optimizations require sudo access..."
    configure_system_performance
    setup_network_interface_optimizations
    
    # Advanced installations (optional)
    read -p "Install DPDK for kernel bypass networking? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_dpdk_dependencies
        setup_huge_pages
    fi
    
    read -p "Install RDMA for direct memory access? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_rdma_dependencies
    fi
    
    read -p "Configure CPU core isolation? (requires reboot) (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        configure_cpu_isolation
    fi
    
    # Create monitoring tools
    create_performance_monitor
    
    # Final verification
    verify_installation
    
    echo
    echo "======================================"
    log_success "Installation completed successfully!"
    echo "======================================"
    echo
    log_info "Next steps:"
    echo "1. Run './performance_monitor.sh' to check system status"
    echo "2. Test your CSP network with: python3 test_optimizations.py"
    echo "3. If you enabled CPU isolation, reboot your system"
    echo "4. Tune parameters in core/config.py for your use case"
    echo
    log_info "For maximum performance:"
    echo "• Use dedicated hardware (10GbE+ network, NVME storage)"
    echo "• Configure DPDK with appropriate network cards"
    echo "• Monitor performance with provided tools"
    echo "• Consider SmartNIC hardware for ultimate performance"
    echo
}

# Run main function
main "$@"

# ============================================================================
# enhanced_csp/network/scripts/setup_dpdk.sh
#!/bin/bash
"""
DPDK Environment Setup Script
Configures DPDK for kernel bypass networking.
"""

set -e

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$source_dir/common.sh"

# DPDK configuration
DPDK_VERSION="22.11"
DPDK_DIR="/opt/dpdk"
HUGEPAGE_SIZE="1G"
HUGEPAGE_COUNT=4

setup_dpdk_environment() {
    log_info "Setting up DPDK environment..."
    
    # Check if DPDK is already installed
    if [ -d "$DPDK_DIR" ]; then
        log_info "DPDK already installed at $DPDK_DIR"
    else
        install_dpdk_from_source
    fi
    
    # Setup huge pages
    setup_dpdk_hugepages
    
    # Bind network interfaces
    setup_dpdk_interfaces
    
    # Configure environment variables
    setup_dpdk_environment_vars
    
    log_success "DPDK environment setup complete"
}

install_dpdk_from_source() {
    log_info "Installing DPDK from source..."
    
    # Install build dependencies
    case $OS in
        "debian")
            sudo apt-get install -y \
                meson ninja-build \
                libnuma-dev libpcap-dev \
                python3-pyelftools
            ;;
        "redhat")
            sudo yum install -y \
                meson ninja-build \
                numactl-devel libpcap-devel \
                python3-pyelftools
            ;;
    esac
    
    # Download and build DPDK
    cd /tmp
    wget "http://fast.dpdk.org/rel/dpdk-${DPDK_VERSION}.tar.xz"
    tar xf "dpdk-${DPDK_VERSION}.tar.xz"
    cd "dpdk-${DPDK_VERSION}"
    
    # Configure build
    meson build
    cd build
    ninja
    
    # Install
    sudo ninja install
    sudo mkdir -p "$DPDK_DIR"
    sudo cp -r ../build/* "$DPDK_DIR/"
    
    log_success "DPDK compiled and installed"
}

setup_dpdk_hugepages() {
    log_info "Setting up DPDK huge pages..."
    
    # Calculate huge page requirements
    local total_memory=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local hugepage_size_kb=$((${HUGEPAGE_SIZE%G} * 1024 * 1024))
    local recommended_count=$((total_memory / hugepage_size_kb / 4))  # Use 1/4 of memory
    
    if [ $recommended_count -lt $HUGEPAGE_COUNT ]; then
        HUGEPAGE_COUNT=$recommended_count
        log_warning "Reduced huge page count to $HUGEPAGE_COUNT based on available memory"
    fi
    
    # Configure huge pages
    echo $HUGEPAGE_COUNT | sudo tee /sys/kernel/mm/hugepages/hugepages-${hugepage_size_kb}kB/nr_hugepages
    
    # Mount huge page filesystem
    sudo mkdir -p /dev/hugepages
    sudo mount -t hugetlbfs nodev /dev/hugepages -o pagesize=${HUGEPAGE_SIZE}
    
    # Make persistent
    echo "nodev /dev/hugepages hugetlbfs pagesize=${HUGEPAGE_SIZE} 0 0" | sudo tee -a /etc/fstab
    
    log_success "Huge pages configured: ${HUGEPAGE_COUNT} x ${HUGEPAGE_SIZE}"
}

setup_dpdk_interfaces() {
    log_info "Setting up DPDK network interfaces..."
    
    # Detect compatible network interfaces
    local interfaces=$(lspci | grep -i ethernet | awk '{print $1}')
    
    log_info "Available network interfaces:"
    for iface in $interfaces; do
        local driver=$(lspci -k -s $iface | grep "Kernel driver" | awk '{print $5}')
        echo "  $iface: $driver"
    done
    
    # Ask user which interface to bind to DPDK
    echo
    read -p "Enter PCI address of interface to bind to DPDK (or 'skip'): " pci_addr
    
    if [ "$pci_addr" != "skip" ] && [ -n "$pci_addr" ]; then
        # Load DPDK kernel modules
        sudo modprobe uio
        sudo modprobe uio_pci_generic
        
        # Bind interface to DPDK
        sudo "$DPDK_DIR/usertools/dpdk-devbind.py" --bind=uio_pci_generic "$pci_addr"
        
        log_success "Interface $pci_addr bound to DPDK"
    else
        log_info "Skipping interface binding"
    fi
}

setup_dpdk_environment_vars() {
    log_info "Setting up DPDK environment variables..."
    
    cat << EOF | sudo tee /etc/profile.d/dpdk.sh
# DPDK Environment Variables
export RTE_SDK="$DPDK_DIR"
export RTE_TARGET="x86_64-native-linux-gcc"
export PKG_CONFIG_PATH="\$PKG_CONFIG_PATH:$DPDK_DIR/lib/x86_64-linux-gnu/pkgconfig"
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:$DPDK_DIR/lib/x86_64-linux-gnu"
EOF
    
    source /etc/profile.d/dpdk.sh
    log_success "DPDK environment variables configured"
}

# Create DPDK test application
create_dpdk_test() {
    log_info "Creating DPDK test application..."
    
    cat << 'EOF' > dpdk_test.py
#!/usr/bin/env python3
"""
DPDK Test Application for CSP Network
Tests DPDK functionality and performance.
"""

import sys
import time
import ctypes
from ctypes import CDLL, c_int, c_char_p

def test_dpdk_environment():
    """Test DPDK environment setup."""
    print("=== DPDK Environment Test ===")
    
    # Check environment variables
    import os
    rte_sdk = os.getenv('RTE_SDK')
    if rte_sdk:
        print(f"✅ RTE_SDK: {rte_sdk}")
    else:
        print("❌ RTE_SDK not set")
        return False
    
    # Check huge pages
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            if 'HugePages_Total' in meminfo:
                huge_total = [line for line in meminfo.split('\n') if 'HugePages_Total' in line][0]
                print(f"✅ {huge_total}")
            else:
                print("❌ Huge pages not configured")
                return False
    except Exception as e:
        print(f"❌ Error checking huge pages: {e}")
        return False
    
    # Check DPDK library
    try:
        lib_path = f"{rte_sdk}/lib/x86_64-linux-gnu/librte_eal.so"
        dpdk_lib = CDLL(lib_path)
        print("✅ DPDK library loadable")
    except Exception as e:
        print(f"⚠️  DPDK library not found: {e}")
    
    return True

def test_network_interfaces():
    """Test DPDK network interface binding."""
    print("\n=== Network Interface Test ===")
    
    import subprocess
    
    try:
        # Run dpdk-devbind.py to show status
        result = subprocess.run([
            "/opt/dpdk/usertools/dpdk-devbind.py", "--status"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("DPDK Interface Status:")
            print(result.stdout)
        else:
            print(f"❌ Failed to get interface status: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error checking interfaces: {e}")

if __name__ == "__main__":
    print("DPDK Test Application")
    print("=" * 40)
    
    env_ok = test_dpdk_environment()
    test_network_interfaces()
    
    if env_ok:
        print("\n✅ DPDK environment appears to be working!")
        print("You can now use DPDK with the CSP network.")
    else:
        print("\n❌ DPDK environment has issues.")
        print("Please check the installation and configuration.")
EOF
    
    chmod +x dpdk_test.py
    log_success "DPDK test application created: ./dpdk_test.py"
}

# Main DPDK setup
main() {
    echo "======================================"
    echo "DPDK Setup for CSP Network"
    echo "======================================"
    echo
    
    check_root
    detect_os
    
    setup_dpdk_environment
    create_dpdk_test
    
    echo
    log_success "DPDK setup completed!"
    echo
    log_info "Test DPDK installation with: ./dpdk_test.py"
    log_warning "Reboot recommended to ensure all changes take effect"
}

main "$@"

# ============================================================================
# enhanced_csp/network/scripts/common.sh
#!/bin/bash
"""
Common functions for CSP network setup scripts.
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root for security reasons."
        log_info "Some commands will use sudo when needed."
        exit 1
    fi
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            OS="debian"
        elif [ -f /etc/redhat-release ]; then
            OS="redhat"
        else
            OS="linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        OS="unknown"
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ============================================================================
# enhanced_csp/network/scripts/benchmark_system.sh
#!/bin/bash
"""
System Benchmark Script for CSP Network
Measures baseline and optimized performance.
"""

set -e

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$source_dir/common.sh"

# Run comprehensive system benchmark
run_system_benchmark() {
    log_info "Running comprehensive system benchmark..."
    
    echo "=== System Information ==="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "OS: $(uname -a)"
    echo "CPU: $(lscpu | grep 'Model name' | awk -F: '{print $2}' | xargs)"
    echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
    echo "Network: $(ip route | grep default | awk '{print $5}' | head -1)"
    echo
    
    # CPU benchmark
    benchmark_cpu
    
    # Memory benchmark  
    benchmark_memory
    
    # Network benchmark
    benchmark_network
    
    # Disk I/O benchmark
    benchmark_disk
    
    # Python performance
    benchmark_python
}

benchmark_cpu() {
    echo "=== CPU Benchmark ==="
    
    # CPU cores and frequency
    echo "CPU cores: $(nproc)"
    echo "CPU frequency: $(lscpu | grep 'CPU MHz' | awk '{print $3}' | head -1) MHz"
    
    # CPU stress test
    log_info "Running CPU stress test (30 seconds)..."
    if command_exists stress; then
        stress --cpu $(nproc) --timeout 30s > /dev/null 2>&1 &
        local stress_pid=$!
        
        # Monitor CPU usage
        for i in {1..6}; do
            local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
            echo "CPU usage sample $i: ${cpu_usage}%"
            sleep 5
        done
        
        wait $stress_pid
    else
        echo "stress tool not available, skipping CPU stress test"
    fi
    
    echo
}

benchmark_memory() {
    echo "=== Memory Benchmark ==="
    
    # Memory information
    free -h
    echo
    
    # Memory bandwidth test
    if command_exists sysbench; then
        log_info "Running memory bandwidth test..."
        sysbench memory --memory-block-size=1M --memory-total-size=10G run
    else
        echo "sysbench not available, skipping memory benchmark"
    fi
    
    echo
}

benchmark_network() {
    echo "=== Network Benchmark ==="
    
    # Network interface information
    local primary_iface=$(ip route | grep default | awk '{print $5}' | head -1)
    if [ -n "$primary_iface" ]; then
        echo "Primary interface: $primary_iface"
        ethtool "$primary_iface" 2>/dev/null | grep Speed || echo "Speed: Unknown"
        
        # Interface statistics
        echo "Interface statistics:"
        cat "/sys/class/net/$primary_iface/statistics/rx_bytes" | awk '{print "RX bytes: " $1}'
        cat "/sys/class/net/$primary_iface/statistics/tx_bytes" | awk '{print "TX bytes: " $1}'
    fi
    
    # Network latency test
    log_info "Testing network latency to common hosts..."
    for host in 8.8.8.8 1.1.1.1 google.com; do
        echo -n "$host: "
        ping -c 3 -W 2 "$host" 2>/dev/null | grep "avg" | awk -F'/' '{print $5}' | awk '{print $1 " ms avg"}' || echo "unreachable"
    done
    
    echo
}

benchmark_disk() {
    echo "=== Disk I/O Benchmark ==="
    
    # Disk space
    df -h | grep -E '^/dev/'
    echo
    
    # Simple disk speed test
    log_info "Running disk I/O test..."
    local test_file="/tmp/csp_disk_test"
    
    # Write test
    echo -n "Sequential write: "
    dd if=/dev/zero of="$test_file" bs=1M count=1000 oflag=direct 2>&1 | grep "copied" | awk '{print $10 " " $11}'
    
    # Read test  
    echo -n "Sequential read: "
    dd if="$test_file" of=/dev/null bs=1M iflag=direct 2>&1 | grep "copied" | awk '{print $10 " " $11}'
    
    # Cleanup
    rm -f "$test_file"
    
    echo
}

benchmark_python() {
    echo "=== Python Performance Benchmark ==="
    
    python3 << 'EOF'
import time
import json
import sys

def benchmark_json_serialization():
    """Benchmark JSON serialization."""
    data = {"test": "data", "numbers": list(range(1000)), "nested": {"key": "value"}}
    
    # Standard JSON
    start = time.perf_counter()
    for _ in range(1000):
        serialized = json.dumps(data)
        json.loads(serialized)
    json_time = time.perf_counter() - start
    
    print(f"Standard JSON (1000 ops): {json_time:.3f} seconds")
    
    # Test orjson if available
    try:
        import orjson
        start = time.perf_counter()
        for _ in range(1000):
            serialized = orjson.dumps(data)
            orjson.loads(serialized)
        orjson_time = time.perf_counter() - start
        
        speedup = json_time / orjson_time
        print(f"orjson (1000 ops): {orjson_time:.3f} seconds ({speedup:.1f}x faster)")
    except ImportError:
        print("orjson not available")

def benchmark_compression():
    """Benchmark compression performance."""
    data = b"Hello, world! " * 1000
    
    import zlib
    start = time.perf_counter()
    for _ in range(100):
        compressed = zlib.compress(data)
        zlib.decompress(compressed)
    zlib_time = time.perf_counter() - start
    
    print(f"zlib compression (100 ops): {zlib_time:.3f} seconds")
    
    # Test LZ4 if available
    try:
        import lz4.block
        start = time.perf_counter()
        for _ in range(100):
            compressed = lz4.block.compress(data)
            lz4.block.decompress(compressed)
        lz4_time = time.perf_counter() - start
        
        speedup = zlib_time / lz4_time
        print(f"LZ4 compression (100 ops): {lz4_time:.3f} seconds ({speedup:.1f}x faster)")
    except ImportError:
        print("LZ4 not available")

def benchmark_async():
    """Benchmark async performance."""
    import asyncio
    
    async def simple_coroutine():
        await asyncio.sleep(0)
        return 42
    
    async def run_async_test():
        start = time.perf_counter()
        tasks = [simple_coroutine() for _ in range(1000)]
        await asyncio.gather(*tasks)
        return time.perf_counter() - start
    
    # Test standard asyncio
    async_time = asyncio.run(run_async_test())
    print(f"asyncio (1000 coroutines): {async_time:.3f} seconds")
    
    # Test uvloop if available
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        uvloop_time = asyncio.run(run_async_test())
        
        speedup = async_time / uvloop_time
        print(f"uvloop (1000 coroutines): {uvloop_time:.3f} seconds ({speedup:.1f}x faster)")
    except ImportError:
        print("uvloop not available")

print("Python version:", sys.version.split()[0])
print()

benchmark_json_serialization()
print()

benchmark_compression()
print()

benchmark_async()
EOF
    
    echo
}

# Generate benchmark report
generate_report() {
    local report_file="csp_benchmark_$(date +%Y%m%d_%H%M%S).txt"
    
    log_info "Generating benchmark report..."
    
    {
        echo "CSP Network System Benchmark Report"
        echo "Generated: $(date)"
        echo "========================================"
        echo
        
        run_system_benchmark
        
    } | tee "$report_file"
    
    log_success "Benchmark report saved: $report_file"
}

# Main benchmark execution
main() {
    echo "======================================"
    echo "CSP Network System Benchmark"
    echo "======================================"
    echo
    
    generate_report
    
    echo
    log_info "Benchmark completed!"
    log_info "Use this baseline to measure optimization improvements"
}

main "$@"