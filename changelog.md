
# Changelog


### 0.3.0

**Major Refactor:**
- Refactored monolithic codebase into focused modules:
  - `actions/` package with individual action files (break, fetch, search, set, timestamp, markdownify)
  - New modules: `execution.py`, `llm.py`, `jinja_utils.py`, `results.py`, `errors.py`, `validation.py`
  - `type_loader.py` and `tools_loader.py` for YAML-based definitions

**New Actions:**
- Web search action (`@search`) using DuckDuckGo
- URL fetching action (`@fetch`) with readability processing
- HTML-to-markdown action (`@markdownify`)
- Timestamp action (`@timestamp`)

**New Features:**
- YAML-based type definitions in `types/` directory (extract, poem, product, speak, superhero, think)
- CLAUDE.md for Claude Code integration guidance
- Comprehensive test coverage for new features

**Syntax Updates:**
- Updated documentation to use `<checkpoint>` syntax instead of `¡OBLIVIATE`

**Dependencies:**
- Added: requests, readability-lxml, markdownify, validators, ddgs
- Optional: playwright for enhanced web fetching


### 0.1.8 (skipped)




### 0.1.7

- Remove chatter CLI tool (use `sd chat` instead)
- Update llm config on response models 
- Improve handling of stdout/stderr streams and progress bars


### 0.1.6

**CLI Improvements:**
- Added `--version` / `-v` flag to show version number (reads from package metadata via importlib)
- Added progress bars to `sd batch` with ETA, percentage, and spinner (using rich library)
- Progress bars automatically hide when stdout/stderr is piped or redirected (TTY detection)
- Added `--quiet` / `-q` flag to suppress progress output while keeping errors visible
- Removed legacy `chatter` CLI tool (use `sd chat` instead)
- All output now follows CLI best practices:
  - **stdout**: Primary output (results, JSON, `--version`, `--help`)
  - **stderr**: Diagnostics (errors, progress bars, verbose logs, warnings)

**Core Features:**
- Added `sd batch` and `sd chat` commands
- Better batch processing with JSON input/output chaining
- Added temporal types and number validation (float, int, min/max constraints)
- Date pattern expansion: "first 2 Tuesdays in September" automatically expands using RRULE
- List extraction with quantifiers: `[[number*:values]]`, `[[number{3}:rgb]]`
- Rationalised required flag handling and !prefix for required fields
- Added LLMConfig class for LLM parameter defaults
- Cache Management Improvements: STRUCKDOWN_CACHE_SIZE environment variable, Automatic LRU eviction when cache exceeds limit
- LLM Configuration - Per-response-type temperature defaults:
    - ExtractedResponse: 0.0 (deterministic)
    - InternalThoughtsResponse: 0.5 (moderate creativity)
    - SpokenResponse: 0.8 (natural variation)
    - DefaultResponse: 0.7 (balanced)
    - Support for model and temperature overrides per response type

