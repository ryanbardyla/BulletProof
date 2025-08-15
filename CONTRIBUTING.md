# Contributing to NeuronLang

Thank you for your interest in contributing to NeuronLang! We welcome contributions that advance biological neural computing.

## Code of Conduct

We are committed to fostering an inclusive and respectful community. Please be kind, constructive, and professional in all interactions.

## How to Contribute

### 1. Fork the Repository
```bash
git clone https://github.com/ryanbardyla/BulletProof.git
cd BulletProof
```

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes

#### Code Standards
- Write clean, documented Rust code
- Follow existing patterns and conventions
- Add tests for new functionality
- Ensure all tests pass: `cargo test --release`

#### Performance Standards
- All claims must be measured, not projected
- Benchmarks must be reproducible
- Document performance implications

### 4. Test Your Changes
```bash
# Run all tests
cargo test --release

# Run benchmarks
cd demos/continual_learning
python3 real_benchmark.py

# Check code formatting
cargo fmt --check
cargo clippy
```

### 5. Submit a Pull Request

Include in your PR:
- Clear description of changes
- Performance impact (if any)
- Test results
- Documentation updates

## Development Guidelines

### Core Principles
1. **Biological Accuracy**: Implementations should reflect real neuroscience
2. **Transparency**: All performance claims must be verifiable
3. **Production Quality**: Code should be ready for real-world use
4. **Energy Efficiency**: Maintain focus on sparse, efficient computation

### Key Areas for Contribution

#### 🧬 Biological Computing
- Protein synthesis improvements
- Synaptic plasticity models
- Neurotransmitter dynamics

#### ⚡ Performance
- GPU optimization
- Sparse matrix operations
- Memory efficiency

#### 📚 Documentation
- Tutorial improvements
- API documentation
- Research papers

#### 🧪 Testing
- Additional benchmarks
- Cross-platform testing
- Edge case coverage

## Testing Requirements

All contributions must:
1. Pass existing tests
2. Include new tests for new features
3. Maintain or improve performance
4. Be reproducible

```bash
# Required checks before submitting
cargo fmt
cargo clippy -- -D warnings
cargo test --release
cargo bench
```

## Documentation

- Update relevant documentation
- Add inline code comments
- Update README if needed
- Document breaking changes

## Performance Benchmarking

For performance-related changes:
1. Run before/after benchmarks
2. Document improvements
3. Ensure reproducibility

```bash
# Benchmark template
cd demos/continual_learning
python3 real_benchmark.py > before.txt
# Make your changes
python3 real_benchmark.py > after.txt
diff before.txt after.txt
```

## Questions?

- Open an issue for bugs
- Start a discussion for features
- Contact: [@ryanbardyla](https://github.com/ryanbardyla)

## Recognition

Contributors will be acknowledged in:
- Release notes
- Documentation
- Research papers (for significant contributions)

Thank you for helping advance biological neural computing!

---

*"Together, we're building the future of AI that learns like the brain."*