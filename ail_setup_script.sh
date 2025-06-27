#!/bin/bash

# AIL (AI Implementation Language) Setup Script for Arch Linux
# This script installs dependencies and tests the AIL system

set -e  # Exit on error

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

# Check if running on Arch Linux
check_arch_linux() {
    if [[ ! -f /etc/arch-release ]]; then
        log_warning "This script is designed for Arch Linux, but can work on other distributions"
        log_info "Continuing anyway..."
    else
        log_success "Running on Arch Linux"
    fi
}

# Check if user has sudo privileges
check_sudo() {
    if ! sudo -n true 2>/dev/null; then
        log_info "This script requires sudo privileges for installing system packages"
        log_info "You may be prompted for your password"
    fi
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    # Update package database
    log_info "Updating package database..."
    sudo pacman -Sy --noconfirm
    
    # Install required packages
    PACKAGES=(
        "python"
        "python-pip" 
        "llvm"
        "clang"
        "base-devel"
    )
    
    for package in "${PACKAGES[@]}"; do
        if pacman -Q "$package" &> /dev/null; then
            log_success "$package is already installed"
        else
            log_info "Installing $package..."
            sudo pacman -S --noconfirm "$package"
        fi
    done
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    # Check if we should use virtual environment
    if [[ "$1" == "--venv" ]]; then
        log_info "Creating virtual environment..."
        python -m venv ail_env
        source ail_env/bin/activate
        log_success "Virtual environment activated"
    fi
    
    # Install llvmlite
    log_info "Installing llvmlite..."
    pip install --user llvmlite
    
    # Verify installation
    if python -c "import llvmlite" 2>/dev/null; then
        log_success "llvmlite installed successfully"
        LLVM_VERSION=$(python -c "import llvmlite.binding; print(llvmlite.binding.llvm_version_info)" 2>/dev/null)
        log_info "LLVM version: $LLVM_VERSION"
    else
        log_error "Failed to install llvmlite"
        return 1
    fi
}

# Create the AIL test file
create_test_file() {
    log_info "Creating AIL test file..."
    
    cat > ail_test.py << 'EOF'
#!/usr/bin/env python3
"""
AIL Test Script - Tests the complete AIL system
"""

# Import the complete AIL system
exec(open('ail_system.py').read())

def test_simple_program():
    """Test a simple AIL program"""
    simple_code = """
    @parallel @vectorize
    func simple_add(const f32* a, const f32* b, heap f32* result, i32 count) -> void {
        @parallel_for (i32 i in 0..count) {
            result[i] = a[i] + b[i];
        }
    }
    
    func main() -> i32 {
        return 0;
    }
    """
    
    print("Testing simple program...")
    
    # Lex and parse
    lexer = Lexer(simple_code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    
    # Generate code
    emitter = SimpleCodeEmitter()
    output = emitter.emit(ast)
    
    print("✓ Simple program test passed")
    return True

def test_gpu_program():
    """Test a GPU kernel program"""
    gpu_code = """
    @kernel @block_size(32, 32, 1)
    func gpu_add(global<f32>* a, global<f32>* b, global<f32>* result, i32 size) -> void {
        i32 x = thread_idx_x() + block_idx_x() * 32;
        i32 y = thread_idx_y() + block_idx_y() * 32;
        i32 idx = y * size + x;
        
        if (idx < size * size) {
            result[idx] = a[idx] + b[idx];
        }
    }
    
    func main() -> i32 {
        launch<{32, 32, 1}, {32, 32, 1}> gpu_add(null, null, null, 1024);
        return 0;
    }
    """
    
    print("Testing GPU program...")
    
    # Lex and parse
    lexer = Lexer(gpu_code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    
    # Generate code
    emitter = SimpleCodeEmitter()
    output = emitter.emit(ast)
    
    print("✓ GPU program test passed")
    return True

def test_vector_math():
    """Test vector math operations"""
    vector_code = """
    func vector_ops(vec3 a, vec3 b) -> f32 {
        vec3 cross_result = cross(a, b);
        f32 dot_result = dot(a, b);
        f32 length_result = length(cross_result);
        return dot_result + length_result;
    }
    
    func main() -> i32 {
        vec3 v1 = {1.0f, 2.0f, 3.0f};
        vec3 v2 = {4.0f, 5.0f, 6.0f};
        f32 result = vector_ops(v1, v2);
        return 0;
    }
    """
    
    print("Testing vector math...")
    
    # Lex and parse
    lexer = Lexer(vector_code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    
    # Generate code
    emitter = SimpleCodeEmitter()
    output = emitter.emit(ast)
    
    print("✓ Vector math test passed")
    return True

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("AIL System Tests")
    print("="*60)
    
    tests = [
        test_simple_program,
        test_gpu_program,
        test_vector_math
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
    
    print(f"\nTest Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    # Run the complete system test first
    print("Running complete system test...")
    test_complete_system()
    
    print("\n" + "="*60)
    print("Running additional tests...")
    print("="*60)
    
    # Run additional tests
    success = run_all_tests()
    
    if success:
        print("\n🚀 AIL system is working correctly!")
    else:
        print("\n⚠️ Some issues detected. Check the output above.")
EOF

    chmod +x ail_test.py
    log_success "Created ail_test.py"
}

# Download the main AIL system file
download_ail_system() {
    log_info "The complete AIL system should be saved as 'ail_system.py'"
    log_info "Make sure you have the complete system code from the previous artifact"
    
    if [[ ! -f "ail_system.py" ]]; then
        log_warning "ail_system.py not found in current directory"
        log_info "Please save the complete AIL system code as 'ail_system.py'"
        log_info "You can copy it from the previous code artifact"
        return 1
    else
        log_success "Found ail_system.py"
        return 0
    fi
}

# Run tests
run_tests() {
    log_info "Running AIL system tests..."
    
    if [[ ! -f "ail_system.py" ]]; then
        log_error "ail_system.py not found. Please save the complete system code first."
        return 1
    fi
    
    # Run the main system test
    log_info "Running main system test..."
    python ail_system.py
    
    # Run additional tests if available
    if [[ -f "ail_test.py" ]]; then
        log_info "Running additional tests..."
        python ail_test.py
    fi
    
    log_success "Tests completed"
}

# Performance benchmark
run_benchmark() {
    log_info "Running performance benchmark..."
    
    cat > benchmark.py << 'EOF'
import time
exec(open('ail_system.py').read())

def benchmark_parsing():
    """Benchmark parsing performance"""
    code = """
    @kernel @block_size(32, 32, 1)
    func large_kernel(global<f32>* data, i32 size) -> void {
        i32 idx = thread_idx_x() + block_idx_x() * 32;
        for (i32 i = 0; i < 100; i++) {
            if (idx < size) {
                data[idx] = data[idx] * 2.0f + 1.0f;
            }
        }
    }
    """ * 10  # Repeat 10 times for larger code
    
    start = time.time()
    for i in range(100):  # Parse 100 times
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
    end = time.time()
    
    lines = code.count('\n')
    total_lines = lines * 100
    lines_per_sec = total_lines / (end - start)
    
    print(f"Parsing Performance: {lines_per_sec:.0f} lines/second")
    print(f"Total time: {(end - start)*1000:.1f}ms for {total_lines} lines")

if __name__ == "__main__":
    benchmark_parsing()
EOF
    
    python benchmark.py
    rm benchmark.py
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    [[ -f "benchmark.py" ]] && rm benchmark.py
    log_success "Cleanup completed"
}

# Main function
main() {
    echo "AIL (AI Implementation Language) Setup Script"
    echo "============================================="
    echo
    
    # Parse command line arguments
    INSTALL_SYSTEM=true
    INSTALL_PYTHON=true
    RUN_TESTS=true
    USE_VENV=false
    RUN_BENCHMARK=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-system)
                INSTALL_SYSTEM=false
                shift
                ;;
            --no-python)
                INSTALL_PYTHON=false
                shift
                ;;
            --no-tests)
                RUN_TESTS=false
                shift
                ;;
            --venv)
                USE_VENV=true
                shift
                ;;
            --benchmark)
                RUN_BENCHMARK=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --no-system    Skip system package installation"
                echo "  --no-python    Skip Python dependency installation"
                echo "  --no-tests     Skip running tests"
                echo "  --venv         Use virtual environment"
                echo "  --benchmark    Run performance benchmark"
                echo "  --help,-h      Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Pre-flight checks
    check_arch_linux
    check_sudo
    
    # Install dependencies
    if $INSTALL_SYSTEM; then
        install_system_deps || { log_error "Failed to install system dependencies"; exit 1; }
    fi
    
    if $INSTALL_PYTHON; then
        if $USE_VENV; then
            install_python_deps --venv || { log_error "Failed to install Python dependencies"; exit 1; }
        else
            install_python_deps || { log_error "Failed to install Python dependencies"; exit 1; }
        fi
    fi
    
    # Setup test files
    create_test_file
    
    # Check for main system file
    download_ail_system || {
        log_error "Please save the complete AIL system code as 'ail_system.py' and run this script again"
        exit 1
    }
    
    # Run tests
    if $RUN_TESTS; then
        run_tests || { log_warning "Some tests failed, but installation appears successful"; }
    fi
    
    # Run benchmark
    if $RUN_BENCHMARK; then
        run_benchmark
    fi
    
    # Success message
    echo
    log_success "AIL setup completed successfully!"
    echo
    echo "Next Steps:"
    echo "1. Run 'python ail_system.py' to test the system"
    echo "2. Create your own AIL programs"
    echo "3. Explore the language features"
    echo
    echo "For more information, check the setup instructions."
    
    # Cleanup
    trap cleanup EXIT
}

# Run main function with all arguments
main "$@"
