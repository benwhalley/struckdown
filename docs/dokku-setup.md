# Dokku Deployment Setup

## Initial Setup

```bash
dokku buildpacks:add sd https://github.com/heroku/heroku-buildpack-python
dokku config:set sd DEFAULT_LLM="gpt-4.1-mini" --no-restart
```

## Nginx Configuration

```bash
dokku nginx:set sd client-max-body-size 50m
dokku nginx:set sd proxy-buffering off
dokku nginx:set sd proxy-read-timeout 600s
dokku proxy:build-config sd
```

## Persistent Storage

The prompt cache needs persistent storage to survive container restarts.

```bash
dokku storage:ensure-directory sd
dokku storage:mount sd /var/lib/dokku/data/storage/sd/prompts:/app/.struckdown/prompts
dokku ps:rebuild sd
```

## Cache Configuration (optional)

```bash
# Max single prompt size (default 1MB)
dokku config:set sd STRUCKDOWN_PROMPT_MAX_SIZE=1048576

# Max total cache size (default 100MB, 0=unlimited)
dokku config:set sd STRUCKDOWN_PROMPT_CACHE_MAX_SIZE=104857600

# Prompt save rate limit (default 10/minute)
dokku config:set sd STRUCKDOWN_PROMPT_RATE_LIMIT=10/minute

# Prompt load rate limit (default 6/minute, prevents hash enumeration)
dokku config:set sd STRUCKDOWN_PROMPT_LOAD_RATE_LIMIT=6/minute
```

## Scheduled Cache Cleanup

Cron is defined in `app.json` -- runs hourly via Dokku's built-in cron.

```bash
# List scheduled tasks
dokku cron:list sd

# Run manually
dokku cron:run sd <cron_id>
# or
dokku run sd python scripts/cleanup_cache.py
```

## Verify Setup

```bash
# Check storage is mounted
dokku storage:list sd

# Check cache stats
dokku run sd python -c "from struckdown.playground.prompt_cache import get_cache_stats; print(get_cache_stats())"
```
