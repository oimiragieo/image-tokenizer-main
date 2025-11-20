# Contributing to Cosmos Image Tokenizer

We welcome contributions! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/cosmos-image-tokenizer.git
   cd cosmos-image-tokenizer
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks:
```bash
black cosmos_tokenizer/
isort cosmos_tokenizer/
flake8 cosmos_tokenizer/
mypy cosmos_tokenizer/
```

## Testing

Write tests for all new features:
```bash
pytest tests/ -v --cov=cosmos_tokenizer
```

Ensure coverage stays above 80%.

## Pull Request Process

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes
3. Add tests
4. Run code quality checks
5. Commit with clear messages:
   ```bash
   git commit -m "Add feature X: description"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Open a Pull Request

## Commit Message Guidelines

Format:
```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Example:
```
feat: Add object-aware compression module

Implements AdaTok-style object-level token merging using SAM.
Reduces token count by 90% while maintaining 96% quality.

Closes #123
```

## Areas for Contribution

### High Priority
- [ ] Pre-trained model checkpoints
- [ ] Object-aware compression (AdaTok/CORE)
- [ ] Dual-codebook architecture (GloTok)
- [ ] Comprehensive benchmarks

### Medium Priority
- [ ] Additional quantization methods
- [ ] Training on more datasets
- [ ] Performance optimizations
- [ ] Better documentation

### Low Priority
- [ ] Mobile deployment
- [ ] ONNX export
- [ ] Additional examples

## Questions?

Open an issue or start a discussion on GitHub.

Thank you for contributing!
