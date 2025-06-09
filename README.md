#  Every Other Token

**A real-time LLM stream interceptor for token-level interaction research**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI API](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/api/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

##  What is this?

Every Other Token is a research tool that intercepts OpenAI's streaming API responses and applies transformations to alternating tokens in real-time. Instead of waiting for complete responses, it intervenes at the token level, creating a new paradigm for LLM interaction and analysis.

##  Why does this matter?

This tool opens up novel research possibilities:

- ** Token Dependency Analysis**: Study how LLMs handle disrupted token sequences
- ** Interpretability Research**: Understand token-level dependencies and causality
- ** Creative AI Interaction**: Build co-creative systems with human-AI token collaboration
- ** Real-time LLM Steering**: Develop new prompt engineering techniques
- ** Stream Manipulation**: Explore how semantic meaning degrades with token alterations

##  Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/every-other-token.git
cd every-other-token

# Install dependencies
pip install openai

# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

### Basic Usage

```bash
# Simple example
python every_other_token.py "Tell me a story about a robot"

# With specific transformation
python every_other_token.py "Explain quantum physics" uppercase

# With different model
python every_other_token.py "Write a haiku" mock gpt-4
```

##  How it works

The script intercepts the OpenAI streaming API response and applies transformations based on token position:

- **Even tokens** (0, 2, 4, 6...): Passed through unchanged
- **Odd tokens** (1, 3, 5, 7...): Transformed using the selected method

### Example Output

**Original**: "The quick brown fox jumps over the lazy dog"
**With `reverse` transform**: "The kciuq brown xof jumps revo the yzal dog"

##  Available Transformations

| Transform | Description | Example |
|-----------|-------------|---------|
| `reverse` | Reverses odd tokens | "hello" → "olleh" |
| `uppercase` | Converts odd tokens to uppercase | "hello" → "HELLO" |
| `mock` | Creates alternating case (mocking text) | "hello" → "hElLo" |
| `noise` | Adds random characters to odd tokens | "hello" → "hello*" |

##  Research Applications

### 1. Token Dependency Studies
```bash
# Study how meaning degrades with token corruption
python every_other_token.py "Solve this math problem: 2+2=" reverse
```

### 2. Semantic Robustness Testing
```bash
# Test how well LLMs maintain coherence under disruption
python every_other_token.py "Continue this story logically..." noise
```

### 3. Creative Collaboration
```bash
# Use transformations to create unexpected creative outputs
python every_other_token.py "Write a poem about nature" mock
```

##  Advanced Usage

### Command Line Arguments

```bash
python every_other_token.py [PROMPT] [TRANSFORM] [MODEL]
```

- `PROMPT`: Your input prompt (required)
- `TRANSFORM`: Transformation type (default: reverse)
- `MODEL`: OpenAI model (default: gpt-3.5-turbo)

### Examples

```bash
# Basic usage
python every_other_token.py "Hello world"

# Specific transform
python every_other_token.py "Explain AI" uppercase

# Different model
python every_other_token.py "Creative writing" mock gpt-4

# All parameters
python every_other_token.py "Technical explanation" noise gpt-4-turbo
```

##  Output Analysis

The tool provides detailed statistics:

```
 EVERY OTHER TOKEN INTERCEPTOR
Transform: reverse
Model: gpt-3.5-turbo
Prompt: Tell me about AI

==================================================
Response (with transformations):

AI si a daorb field fo computer ecneics that...

==================================================
 Complete! Processed 156 tokens.
 Transform applied to 78 tokens.
```

##  Research Ideas

### Immediate Experiments

1. **Causality Testing**: How does corrupting early tokens affect later generation?
2. **Semantic Drift**: At what corruption level does meaning break down?
3. **Model Comparison**: How do different models handle token disruption?
4. **Domain Analysis**: Which topics are most/least robust to token corruption?

### Advanced Research

1. **Recursive Mutation**: Feed transformed output back as input
2. **Multi-Model Chains**: Use tokens from different models alternately
3. **Human-in-the-Loop**: Replace odd tokens with human input
4. **Bidirectional Analysis**: Compare forward vs backward token importance

##  Error Handling

The tool includes comprehensive error handling:

- **API Key Validation**: Checks for valid OpenAI API key
- **Network Error Recovery**: Handles connection issues gracefully
- **Invalid Transform Detection**: Validates transformation types
- **Model Availability**: Checks if requested model exists

##  Contributing

We welcome contributions! Here are ways to get involved:

1. **New Transformations**: Add creative token transformation functions
2. **Analysis Tools**: Build utilities for analyzing output patterns
3. **Visualization**: Create tools to visualize token-level changes
4. **Documentation**: Improve examples and research applications

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/every-other-token.git
cd every-other-token

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

##  Research Papers & Citations

If you use this tool in academic research, please cite:

```bibtex
@software{every_other_token,
  title={Every Other Token: Real-time LLM Stream Interceptor},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/every-other-token}
}
```

## 🔮 Future Roadmap

- [ ] **Web Interface**: Browser-based tool for easier experimentation
- [ ] **Batch Processing**: Process multiple prompts simultaneously
- [ ] **Export Functionality**: Save results in various formats (JSON, CSV)
- [ ] **Visualization Dashboard**: Real-time charts and analysis
- [ ] **Custom Transformations**: User-defined transformation functions
- [ ] **Multi-API Support**: Extend to other LLM providers (Anthropic, Cohere)
- [ ] **Collaborative Mode**: Multiple users contributing tokens
- [ ] **Research Templates**: Pre-built experiments for common research patterns

##  Important Notes

- **API Costs**: Streaming API calls count toward your OpenAI usage
- **Rate Limits**: Respect OpenAI's rate limiting policies
- **Research Ethics**: Consider implications when studying AI behavior
- **Data Privacy**: Be mindful of sensitive information in prompts

##  Related Work

- [Real-time LLM Steering](https://example.com/paper1)
- [Token-level Interpretability](https://example.com/paper2)
- [Adversarial Prompt Engineering](https://example.com/paper3)

##  Known Issues

- Very long responses may hit API timeout limits
- Some Unicode characters may not transform correctly
- Rapid token streaming can occasionally cause display issues

##  License

MIT License - see [LICENSE](LICENSE) file for details.

##  Acknowledgments

- OpenAI for the streaming API
- The AI research community for inspiration
- Contributors and beta testers

##  Support


- **Email**: mattbusel@gmail.com

---

**Made with 🧬 for AI researchers, prompt engineers, and curious minds**

*"Every token tells a story. Every other token tells a different one."*
