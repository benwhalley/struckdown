# TODO

## Grammar: allow hyphens in positional `[[pick:...]]` options

`struckdown/grammar.lark` defines positional pick options via lark's `CNAME`
token (Python-identifier shape: `[_A-Za-z][_A-Za-z0-9]*`). Hyphenated slugs
like `writing-concisely` tokenise as `CNAME('writing')` and then fail on
`-concisely`, breaking templates like
`[[pick:task|writing-concisely,mean-marker,...]]`.

Slugs / kebab-case ids are common, so positional options should accept them
natively. Add a `SLUG` token (`("_"|"-"|LETTER) ("_"|"-"|LETTER|DIGIT)*`) and
extend `positional_value` to emit it as a literal string:

```
positional_value: STRING -> literal_string
                | SIGNED_FLOAT -> literal_float
                | NUMBER -> literal_number
                | EMOJI -> literal_emoji
                | SLUG -> literal_slug   # new -- accepts hyphens
                | CNAME -> literal_cname
```

Workaround until then: quote each option (`[[pick:task|"writing-concisely",...]]`).
psybot/record/ai.py applies this workaround in `match_task2`.
