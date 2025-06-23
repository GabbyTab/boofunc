<p align="center">
  <img src="branding/logos/boo_horizontal.png" alt="BoolFunc Logo" width="927"/>
</p>



<p align="center">
  <a href="https://numfocus.org/"><img src="https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg" alt="Powered by NumFOCUS"></a>
  <a href="https://pypi.org/project/boolfunc/"><img src="https://img.shields.io/pypi/dm/boolfunc.svg?label=PyPI%20downloads" alt="PyPI downloads"></a>
  <a href="https://anaconda.org/conda-forge/boolfunc"><img src="https://img.shields.io/conda/dn/conda-forge/boolfunc.svg?label=Conda%20downloads" alt="Conda downloads"></a>
  <a href="https://stackoverflow.com/questions/tagged/boolfunc"><img src="https://img.shields.io/badge/stackoverflow-ask%20questions-orange.svg" alt="Stack Overflow"></a>
  <a href="https://securityscorecards.dev/viewer/?uri=github.com/username/boolfunc"><img src="https://api.securityscorecards.dev/projects/github.com/GabbyTab/boofunc/badge" alt="OpenSSF Scorecard"></a>
  <a href="https://github.com/GabbyTab/boofunc/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/typing-checked-blue.svg" alt="Typing"></a>
</p>

boolfunc: Comprehensive Boolean Function Analysis Library
How do I include a build satatus?



# 🚀 Features

## Core Capabilities

- **Multiple Representations**  
  Seamless conversion between truth tables, polynomials (ANF), circuits, and spectral forms.

- **Spectral Analysis**  
  Complete Fourier analysis toolkit with influence computation and noise stability.

- **Property Testing**  
  Classical and quantum property testing algorithms with query complexity analysis.

- **Quantum Extensions**  
  Quantum Boolean function analysis and quantum property testing.

- **Advanced Visualization**  
  Interactive plots for spectral properties, influences, and function behavior.

- **High Performance**  
  Optimized implementations using NumPy, SciPy, and optional Numba acceleration.

---

## Built-in Function Library

- Tribes functions
- Majority and threshold functions
- Dictator functions
- Random Boolean functions
- Cryptographic Boolean functions
- Custom function construction tools


# 📦 Installation

## Install from PyPI (Recommended)

```bash
pip install boolfunc
```
From Source
```bash
git clone https://github.com/username/boolfunc.git
cd boolfunc
pip install -e .
```
With Optional Dependencies
```bash

# For visualization
pip install boolfunc[viz]

# For quantum computing features
pip install boolfunc[quantum]

# For development
pip install boolfunc[dev]

# All features
pip install boolfunc[all]
```
# 🏃‍♀️ Quick Start
```python
import boolfunc as bf
import numpy as np

# Create a Boolean function from truth table
truth_table = [0, 1, 1, 0, 1, 0, 0, 1]  # 3-variable function
f = bf.create(truth_table)

# Or create built-in functions
majority = bf.BooleanFunction.majority(3)
tribes = bf.BooleanFunction.tribes(k=2, n=6)

# Seamless representation conversion
polynomial = f.to_polynomial()
circuit = f.to_circuit()

# Spectral analysis
analyzer = bf.SpectralAnalyzer(f)
influences = analyzer.influences()
total_influence = analyzer.total_influence()
noise_stability = analyzer.noise_stability(rho=0.9)

# Fourier analysis
fourier_coeffs = analyzer.fourier_expansion()
spectral_concentration = analyzer.spectral_concentration(degree=2)

# Property testing
tester = bf.PropertyTester(f)
is_linear = tester.blr_linearity_test(num_queries=3)
is_constant = tester.constant_test()

# Visualization
viz = bf.Visualizer(f)
viz.plot_influences()
viz.plot_fourier_spectrum()
viz.plot_noise_stability()

# Quantum extensions (optional)
qf = bf.quantum.QuantumBooleanFunction(f)
quantum_fourier = qf.quantum_fourier_analysis()
```


---

# 📚 Documentation

Comprehensive documentation is available at [boolfunc.readthedocs.io](https://boolfunc.readthedocs.io), including:

- **Getting Started Guide:** Basic concepts and first steps
- **API Reference:** Complete function and class documentation
- **Mathematical Background:** Theory and algorithms explained
- **Advanced Examples:** Research-oriented tutorials
- **Performance Guide:** Optimization tips and benchmarking

**Quick Links:**
- [Installation Guide](https://boolfunc.readthedocs.io/en/latest/installation.html)
- [Tutorial Notebooks](https://boolfunc.readthedocs.io/en/latest/tutorials.html)
- [API Reference](https://boolfunc.readthedocs.io/en/latest/api.html)
- [Mathematical Theory](https://boolfunc.readthedocs.io/en/latest/theory.html)

---

# 🧪 Core Modules

## `boolfunc.core`
- Multiple Boolean function representations with automatic conversion:
  - Truth table representation
  - Polynomial (ANF) representation
  - Circuit representation
  - Spectral representation

## `boolfunc.analysis`
- Comprehensive spectral analysis tools:
  - Fourier expansion and Walsh-Hadamard transforms
  - Variable influences and sensitivity analysis
  - Noise stability and hypercontractivity
  - Boolean convolution operations
  - Spectral concentration measures

## `boolfunc.testing`
- Property testing algorithms:
  - BLR linearity testing
  - Constant function testing
  - Junta testing
  - Monotonicity testing
  - Custom testing framework

## `boolfunc.quantum`
- Quantum computing extensions:
  - Quantum Boolean function analysis
  - Quantum property testing
  - Quantum Fourier analysis
  - Quantum circuit synthesis

## `boolfunc.visualization`
- Advanced plotting capabilities:
  - Influence distribution plots
  - Fourier spectrum visualization
  - Noise stability curves
  - Interactive spectral exploration

---

# 🔬 Research Applications

BoolFunc is designed for researchers in:

- **Theoretical Computer Science:** Analysis of computational complexity
- **Cryptography:** Boolean function cryptanalysis and design
- **Property Testing:** Development of efficient testing algorithms
- **Quantum Computing:** Quantum algorithm analysis and design
- **Machine Learning:** Boolean function learning and optimization
- **Combinatorics:** Extremal problems and probabilistic methods

---

# 📈 Performance

BoolFunc is optimized for both small research examples and large-scale computations:

- **Vectorized Operations:** Efficient NumPy-based implementations
- **JIT Compilation:** Optional Numba acceleration for critical paths
- **Memory Efficiency:** Sparse representations for large functions
- **Parallel Processing:** Multi-core support for independent computations

**Benchmarks:**
```python
# Performance example: 10-variable Boolean function analysis
import boolfunc as bf
import time

# Create random 10-variable function
f = bf.BooleanFunction.random(n_vars=10, seed=42)

# Benchmark spectral analysis
start = time.time()
analyzer = bf.SpectralAnalyzer(f)
influences = analyzer.influences()
fourier_coeffs = analyzer.fourier_expansion()
end = time.time()

print(f"Analysis completed in {end - start:.3f} seconds")
```

# 🤝 Contributing

We welcome contributions from the research community!  
Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Development Setup

```bash
# Clone repository
git clone https://github.com/username/boolfunc.git
cd boolfunc

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
flake8 src/
black src/
mypy src/
```

## How to Contribute

- **Bug Reports:** Use GitHub issues with detailed reproduction steps
- **Feature Requests:** Discuss proposals in GitHub discussions
- **Code Contributions:** Submit pull requests with tests and documentation
- **Documentation:** Help improve docs and add examples
- **Research Integration:** Share your research applications and use cases

---

# 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

# 📞 Support and Community

- **Documentation:** [boolfunc.readthedocs.io](https://boolfunc.readthedocs.io)
- **Issues:** [GitHub Issues](https://github.com/username/boolfunc/issues)
- **Discussions:** [GitHub Discussions](https://github.com/username/boolfunc/discussions)
- **Email:** boolfunc-support@example.com

---

# 🙏 Acknowledgments

BoolFunc builds upon decades of research in Boolean function analysis.  
We acknowledge the foundational work of researchers in:

- Harmonic analysis on Boolean cubes
- Property testing theory
- Quantum computing and quantum query complexity
- Computational complexity theory
- Modern cryptanalysis techniques

---

# 📖 Citation

If you use BoolFunc in your research, please cite:

```text
@software{boolfunc2024,
  title={BoolFunc: A Comprehensive Python Library for Boolean Function Analysis},
  author={Gabriel Taboada},
  year={2024},
  url={https://github.com/username/boolfunc},
  version={1.0.0}
}
```

---

# 🗺️ Roadmap

## Version 1.0 (Current)
- ✅ Core representations and conversions
- ✅ Spectral analysis toolkit
- ✅ Basic property testing
- ✅ Visualization framework

## Version 1.1 (Planned)
- 🔄 Quantum computing extensions
- 🔄 Advanced cryptographic functions
- 🔄 Performance optimizations
- 🔄 Interactive Jupyter widgets

## Version 1.2 (Future)
- 📋 Distributed computing support
- 📋 GPU acceleration
- 📋 Advanced machine learning integration

Happy Boolean function analyzing! 🎯
<p align="left">
  <img src="branding/logos/boo_alt.png" alt="BoolFunc Logo" width="927"/>
</p>


For questions, suggestions, or collaboration opportunities, don't hesitate to reach out through our community channels.
