

some rules about pipelines/dags we should enforce

- no loops
- no [[output]] from reduce nodes


- fix naming of analyses in similarity comparison


- check unzipping better
- allow for max_file sizes by-user
- make credentials per-user
- allow user to add credentials when starting the run (store in localstorage not the db)










in a future version of struckdown we would like to offer 


- chunked responses for prompt segments. i.e. return a genertor of prompt segments as they become available (phase 1)

- streaming responses for each completion (phase 2)


Think about how to do this ina  way that is compatible with the current design and document thoughts and plans in todos/streaming.md







in retrospect maybe doing everyting in memory is foolish?
could we use joblib instead to cache the text extracted from file uploads to disk? we would use a cached function which would return the text and the file name/path. then, when the server tries to use a file again (addressed by uuid) then the cache would be hit. if it's a miss we could raise an error. this way we can use joblib to manache the cache size as we already do for llm results?
plan how this could work? think through the security implications of this approach. think through some sane limits if exposing this to public access




for `sd edit ..`  and maybe for all chatter() calls...
let's use ImmutableSandboxedEnvironment to improve security?
https://jinja.palletsprojects.com/en/stable/sandbox/#jinja2.sandbox.ImmutableSandboxedEnvironment

is there a reason why chatter shoul not always use ImmutableSandboxedEnvironment? would any features break? could we use the intercepted_binops list to detect struckdown syntax that should be intercepted and escaped? would that save doing it with regex as we currnetly do?









 2. Zlib Decompression Bomb - MEDIUM SEVERITY

  Location: struckdown/playground/core.py:131-132

  Issue: zlib.decompress(compressed) on user input with no size limit. A small compressed payload can expand to gigabytes.

  Attack:
  # Create decompression bomb
  import zlib, base64
  bomb = zlib.compress(b'A' * (1024 * 1024 * 500))  # 500MB
  encoded = base64.urlsafe_b64encode(bomb).decode()
  # Visit /e/<encoded> - server OOMs

  Impact: DoS via memory exhaustion.
  
use Zstandard instead? set limit on expanded size?






  ---
  3. Large JSON Parsing - MEDIUM SEVERITY

  Location: All request.get_json() calls in flask_app.py

  Issue: No limit on JSON body size except Flask's MAX_CONTENT_LENGTH (5MB default).

  Attack: Send 5MB JSON with deeply nested structures to cause CPU exhaustion during parsing, or large arrays to exhaust memory.



    
    
    
    
    
 7. Missing CSRF Protection - LOW SEVERITY

  Location: All POST endpoints in flask_app.py

  Issue: No CSRF tokens on forms. State-changing requests can be forged.

  Real Risk: LOW - The app doesn't have user authentication/sessions, so CSRF impact is limited. An attacker could:
  - Execute prompts using victim's API key (if configured in Settings and victim visits malicious site)
  - Upload files on behalf of user
  
  
How to mitigate this? I don't want keys to be abused by malicious sites. could we store a token in localstorage and check it on every request? 