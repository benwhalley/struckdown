
New ideas for feature:


### Date and time response type

look in return_type_models.py and __init__.py and also grammar.lark

The idea is that we want to be able to extract dates, times, datetimes and durations from the text using prompts

we would extend the existing syntax for completions to allow:


[[date:varname]]

[[datetime:varname]]

[[time:varname]]

[[duration:varname]]


Dates would be python datetime objects
Times would be python time objects (from datetime import time)
durations would be python timedelta objects (from datetime import timedelta)

When extracting these values, we always want to give context to the LLM. So we'd include the current date, time, and timezone in the prompt somehow (the user doesn't need to do this by hand.. it happens automatically, or is added as a hint to the field definitions?)

so for example, if the user says "next tuesday" the model can correctly infer actual date relative to the current date

make a plan for how to do this and what tests we need to write



one tweak - let's allow for the model to return a NULL if nothing can be found in the text that converts to a date, time, datetime or duration.
this should be the default behaviour, but we can add an option `required` to make the extraction fail if no valid value is produced:



"sdasdasdasd [[date:varname]]"

default behaviour produces None

"sadadasdsd [[date:varname|required]]"

this behaviuour produces an error



one tweak, let's allow options for these date and time fields.
at present we include the real curtrent date, time, and timezone in the prompt, but the user might want to 'fake' it... let them include a specific current_time, current_date or current_datetime or timezone in the completion syntax.

e.g.

[[date:varname|current=2023-01-01]]


