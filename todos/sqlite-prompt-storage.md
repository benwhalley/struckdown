# Plan: SQLite-based Prompt Storage with Peewee ORM

## Summary

Migrate prompt storage from flat files to SQLite using **Peewee ORM** for declarative models, automatic schema creation, and clean query API. Adds metadata tracking including version lineage via `parent_id`.

**Critical constraint**: Only save prompt text -- NEVER inputs or outputs.

## Current State

The flat-file implementation exists at `struckdown/playground/prompt_cache.py`:
- Stores prompts as `~/.struckdown/prompts/{uuid}.txt`
- Functions: `store_prompt()`, `get_prompt()`, `validate_prompt_id()`, `prompt_exists()`, `get_cache_stats()`

## New Architecture

### Database Location
- Path: `~/.struckdown/prompts.db` (configurable via `STRUCKDOWN_PROMPT_DB`)
- Single file, easy to backup/deploy

### Peewee Model (declarative schema)

```python
from peewee import SqliteDatabase, Model, TextField, DateTimeField, ForeignKeyField
import datetime

db = SqliteDatabase(None)  # deferred init

class Prompt(Model):
    id = TextField(primary_key=True)  # UUID
    parent = ForeignKeyField('self', null=True, backref='children')
    text = TextField()
    created_at = DateTimeField(default=datetime.datetime.utcnow)

    class Meta:
        database = db
        table_name = 'prompts'
```

Peewee auto-creates tables on `db.create_tables([Prompt])`.

### API (same public interface, new implementation)

**Backend (`prompt_cache.py`)**:
- `store_prompt(prompt_id, text, parent_id=None)` -- add optional parent_id
- `get_prompt(prompt_id)` -- returns text (unchanged interface)
- `get_prompt_with_metadata(prompt_id)` -- returns dict with id, text, parent_id, created_at
- `get_prompt_history(prompt_id)` -- returns list of ancestors (for future UI)
- `list_prompts(limit=100, offset=0)` -- paginated list for admin/debug
- `get_cache_stats()` -- count, db file size
- `validate_prompt_id(prompt_id)` -- unchanged
- `prompt_exists(prompt_id)` -- unchanged

**Flask (`flask_app.py`)**:
- `POST /api/save-prompt` -- add optional `parent_id` in request body
- `GET /p/<prompt_id>` -- pass `prompt_id` to template (already done)

**JavaScript (`editor.js`)**:
- Track `currentPromptId` from hidden field
- When saving, include `parent_id: currentPromptId` in request body
- After save, update `currentPromptId` to the new UUID

## Files to Modify

### 0. `pyproject.toml`
Add dependency: `peewee>=3.17.0`

### 1. `struckdown/playground/prompt_cache.py`
Replace flat-file implementation with Peewee/SQLite:
- Define `Prompt` model class
- `_init_db()` initialises connection and creates tables
- Use WAL mode for concurrent access: `db.pragma('journal_mode', 'wal')`
- Maintain same public API for backwards compatibility

### 2. `struckdown/playground/flask_app.py`
- Modify `POST /api/save-prompt` to accept optional `parent_id`
- Validate `parent_id` is a valid UUID if provided

### 3. `struckdown/playground/static/js/editor.js`
- Read `currentPromptId` from hidden field on page load
- Include `parent_id` in save-prompt request body
- Update hidden field after successful save

### 4. `struckdown/tests/test_prompt_cache.py`
- Update tests for SQLite/Peewee backend
- Add tests for parent_id tracking
- Add tests for `get_prompt_history()`

## Implementation Steps

1. Add `peewee>=3.17.0` to pyproject.toml
2. Rewrite `prompt_cache.py` with Peewee model and functions
3. Update Flask endpoint to accept parent_id
4. Update editor.js to track and send parent_id
5. Update tests
6. Delete old `~/.struckdown/prompts/` directory (manual cleanup)

## Migration

No migration needed -- flat file storage not yet deployed. Clean replacement.

## Dokku Persistence

```bash
dokku storage:ensure-directory struckdown
dokku storage:mount struckdown /var/lib/dokku/data/storage/struckdown:/root/.struckdown
```

## Security Considerations

- Peewee uses parameterised queries (SQL injection safe)
- Validate UUIDs before use
- No sensitive data stored (prompt text only)
- Rate limiting: 20/min (already implemented)
- WAL mode for concurrent access reliability
