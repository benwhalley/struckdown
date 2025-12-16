```
<system>
You are an experienced academic, keen to help students with their work.
</system>

# Student work

{{source}}

# Feedback task

Extract the following information about the student essay:

<parallel>
Does it have a clear argument? [[clear_argument]]
Does it have a discuss medical ethics? [[clear_argument]]
Does it have a clear conclusion? [[clear_argument]]
</parallel>


```


I want to introduce a new prompting construct into struckdown.

Normally, each prompt completion is available in context for subsequent slots.
However, in the example above I define a <parallel> tag.

Completions in this parallel tag are not available in context for subsequent slots.

Essentially, within the checkpoint, the contents of the <parallel> tag are split at each [[slot]]
Each slot is triggered in parallel, and the variable saved in context for the start of the next slot.

So, it would be an error to do this

<parallel>
What was the main argument? [[clear_argument]]
{{argument}} does it make sense? [[sense]]
</parallel>

Becasue the variable {{argument}} would not be available in context when we complete [[sense]]



TODO: check the <parallel> tag name is sensible for non technical users
or suggest alternatives


think clearly about how this could be parsed within the existing machinery.
keep it dry. don't recreate existing functionality ... use what we have if we can

ultrathink
