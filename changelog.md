
# Changelog



### 1.1.6

- Added `sd batch` and `sd chat`
- Better batch processing with JSON input/output chaining
- Added temporal types and number validation (float, int, min/max constraints)
- Date pattern expansion: "first 2 Tuesdays in September" automatically expands using RRULE
- List extraction with quantifiers: [[number*:values]], [[number{3}:rgb]]
- Rationalised required flag handling and !prefix for required fields
- Added LLMConfig class for LLM parameter defaults
- Cache Management Improvements: STRUCKDOWN_CACHE_SIZE environment variable,  Automatic LRU eviction when cache exceeds limit
- LLM Configuration.  Per-response-type temperature defaults:
    - ExtractedResponse: 0.0 (deterministic)
    - InternalThoughtsResponse: 0.5 (moderate creativity)
    - SpokenResponse: 0.8 (natural variation)
    - DefaultResponse: 0.7 (balanced)
    - Support for model and temperature overrides per response type

  