i would like a playground cli tool for struckdown

`sd edit myfile.sd` would fire up a small server (use python - probably flask but see notes below)
this would run on a local port in range 9000+ (if the port was busy then a new one found) it would also open the default web browser to show the UI of the server

the myfile.sd is a struckdown prompt in the local dir
if it's omitted, the cli creates untitled.sd in the cwd  if doesn't exist and edits that

the UI would allow users to 
- define and edit SD syntax
- define inputs to the context, either individually or by uploading a tabular file

then, as they edit the sd syntax, the prompt would be executed with the inputs and display the outputs

the ui would use bootstrap css styling

the UI would present two columns:

- left column would have A large text editing box with struckdown syntax 
- the right column would be for outputs of the model

- additionally, a popover-style interface on the right or top of the page would allow the user to specify either:
    - "inputs" for the prompt in textareas
    - an xlsx file used to do variable completions
    
    
the popover would be a kind of setting panel which could be collapsed or hidden while editing 

- each {{var}} identified in the struckdown  would create a textarea which could be edited 

- variables that match a slot_name in the syntax ... these are created by the model typically, so we don't need to add input for them

- if the user instead wants to define inputs through xlsx (as they might using `sd batch` then they could upload a file in this panel)

- the file and other inputs would persist as the prompt is edited and re-run


The popover would also have options for specifying the LLM model name to use (as a char field)

When the user saves edits the prompt, the outputs are re-rendered when the user hits `save`' Re-rendering updates all the outputs.



### Editor panel


Editor box highlighted using codemirror or similar. there would be a button to save the syntax changes. probably use a textarea with styling?

the whole column should be resizeable. User can make narrower if they wish, or wider if they want to focus on editing.

the editor should use codemirror or the syntax highlighting already provided in the struckdown package. Font should be monospaced and 14pt


### Outputs panel

This shows the results of the prompt outputs. we use bootstrap cards to separate outputs but otherwise keep styling minimal

show the variable name of the slot in `code` style - bold, small, but distinctive.

show the output text in a regular font.

add css classess based on the type of slot so we can style them in future

font should be fairly small

each output should have a checkbox next to it which allows the user to "pin" it at the top of the list. this means that intermediate steps can be shown after the final (perhaps most important) output

otherwise, outputs shown in order of slot names in prompt


If the user has selected to use an xlsx file as input, then
this panel displays tabular output, where each row was an input in the xlsx, and each col relates to a slot name.
the table will allow the user to show input cols, but these are hidden/collapsed by default.
use a javascript table library to display?


### Settings and Inputs panel/popover


this allows user to 

- upload a file OR
- add individual texts to specofy the input for each variable.

the UI shoudl make clear this is an either/or.
maybe use tabs?
persist the inputs they make if they switch from one to the other, but only send the one they are actively using



# Implementation

i'\m flexible, but anticipate us using 

- python server,  to implement the logic and call chatter() using the inputs or use the batch functions to process a file

- a js user interface, handling the syntax highlighting, and specifying the  input slots


When the user updates the prompt, some of the other user interface elements need to change

e.g. when we add a [[slot]] to the syntax, this will update the inputs required.
this should probably happen using partial updates to the page, e.b. via htmx or similar js lib
we definitely don't want to reload the whole page each time because then editor state would be lost and it would intterup the user flow.

similarly, when we press save, we want to outputs to be updated wihtout reloading the whole page

would partial updates like this require we spawn a multithread server? hopefully not. but the server and rendering using chatter needs to be separate from the cli and server thread?

# Actions

When running locally, extra return types and actions can be loaded as when using `sd chat` or `sd batch`


# Future/Phase 2

It would be nice if this UI could be exposed on the web.

- users could add their own LLM API key
this would get saved in localstorage so user doesn't need to edit it twice

- each user session would create a unique url, 

- saving the prompt would base64 encode it into the url? so that way we could share prompts by sharing the url? the model-name and any other setting would also get encoded so sharign the URL really does share evetytning 

- the interface would allow them to save prompt edits and re-render the same way?

- Only predefined actions are allowed in this playground if running remotely?

Don't make decisions now that would prevent phase2 from happening safely and easily


# Pre-existing implenetation

You can look at ~/dev/mindframe for PromptPlaygroundView 
this is a similar concept, but executed as part of a django web app
we probably don't want to tie to a particular django app, although it could potentially be implemented as a pluggable django app where the views are all created and someone just hooks up the urls. for the Command line `sd edit` version we would just bundle a django app?? 


# Approach

make an implementation plan considering all facets of the task
- technologies used
- code organisation
- ui design and organisation

make a decision or recommendations on using flask vs making this a pluggable django application.

prioritise clean resuable design

make the html and css clean and easy to edit. 

use bootstrap in preference as css baseline, but tailwind or others also fine if they are more suitable

use htmx for partial updates unless you can think of a better plan?


### Tests and development

If possible, use test-driven development.
as part of the plan, define a set of tests whcih exercise the core functionality and enable you to work as autonomously as possible. you can use the tests to judge your own progress and only stop editing code nice you have got the tests passing to your own satisfaxction.

examples (not exhaustive) of some tests whic might be needed:

- editor avalable in the page
- save function works and re-renders the prompt
- outputs from the prompt execution are displayed properly
- settings panel allows specifying an xlsx and uploading one induces a batch mode

Some of the tests might require using playwright to test properly


# Using a Database

Avoid using a database if possible. 
When run locally the editor will save changes to the local prompt file.

When running remotely, the editor would save changes to a base64 string encoded in the url and force the browse to update location so this can be copied and pasted to share

when running remotely, we would have to consider batching the execution of promots though... this might require some kind of queue

use redis in preference for the queue — but only if needed?
if it's simpler, just create a /tmp folder which a job list, and make name these with a uuid or a hash of the inputs? then pass the uuid or hash back to the client so it can poll for updates? or some other mechanism, but think clearly about how the data flow will work

in the first instancr kep it simple and allow re-executing to block, but for a remote serveice we'd need to fix this


# Make a plan

ultrathink 

ask me any clarifying questions needed. especially if there appear to be tradeoffs to make or decisions which require taste or judgement (but present the options clearly after thinking them through.)

put your implenenation plan into a new md document for me to review/edit before we get started










--------------------


amends


- confirm using flask. remove django stuff from plans

- use datatable or similar to display batch data right away

- confirm using htmx -  remove deliberation

- separate inputs and setting panels. these can both be collapsed-by-defalt panels


- When xlsx is selected, outputs panel becomes a table:
BUT this may need to include inputs and outputs to identify rows 

- can trhe batch rows be loaded incrementally as they are processed?

- abaondon phased approach - implement both local and remote now. formalise the plan, or do them togehter



## Questions for Clarification

1. **Custom Actions Loading:** When running locally, should `sd edit` look for custom actions in the same way `sd chat` does (via `-I` paths)? Or should it be more restricted?

yes - same as sd chat.
same when running remotely... look in working dir where server is run from


2. **Multiple Files:** Should the UI support switching between multiple .sd files, or is single-file focus sufficient for v1?

single focus 



3. **Output Persistence:** When the user closes and reopens the playground, should previous outputs be restored? This would require storing state somewhere (file or browser localStorage).

we could encode state in the base64 url? this way opening the same url would show the same state?

however, in local mode, we do need to show what file is being edited... so maybe don't need to encoding for that local version?


4. **Syntax Validation:** Should we show real-time parse errors as the user types, similar to how the mindframe implementation validates? Or only on save?


yes please. but don't affect editing. could we show errors in a panel above the editor whcih appear when error triggered?


5. **Resizable Columns:** Is drag-to-resize columns a must-have for Phase 1, or can we start with fixed proportions?

nice to have if not difficult


6. **Export in Single Mode:** Should users be able to export single-mode outputs (copy as JSON, save as markdown), or is that Phase 2?

no - for single mode export not needed. for batch mode would be nice to be able to download the xlsx






this seems great and works at first glance

Review the code written against the plan and check evetything is implemnented
make a new document to check progress aganst the plain.

Especially, Make sure the tests actually ensure all features are present and working Use playwright and browser automation to make it a really thorough test







BUGS/AMENDMENTS

I really like the UI so only minimal change: But, the toggle between single and batch mode should be in the sample place as the 'inputs'
the user should select 'single' and then see the invidivual inputs.
or if the select batch, the view changes to see the file input (instead of the invidiual inputs)
is that clear?






in remote mode ( `sd serve`) we should default to not reading the API key from the environment.
require the user to enter it in the settings panel.
we could have an option to read it from the environment, but that should be explicit and required. e.g. sd serve --api-key=$MY_API_KEY


Add a new feature
In addition to single and batch mode inputs, we should have a file mode

- single means entering text by hand
- batch means uploading an xlsx file, or a zip archive of files (we need to support that too... currently only supports xlsx)
- file mode means uploading a single file


When uploafing a zip or a single file in file mode, the file should be included as {{source}} in the prompt context











flask-limiter

STRUCKDOWN_MAX_SYNTAX_LENGTH = set default quite large... 1m chars

don't worry about batch limits

remote mode should disable search, fetch actions  by default
we should have a property when registering actions to `allow_remote_use` and default to false for these. only set to true for safe actions like set, break

let api reject fake model names


what sorts of input file validation should we do? help me think about that









UI / UX changes
make the settings and inputs panels into tabs
make it so the tab body is collased by default
when you click on either tab, that tab body is expanded
clicking on the same tab again collapses the body
clicking on a different tab swaps to the other tab

color code the tabs... use a nice color for the input and grey for the settings buttons. the bg of the tab body will be grey, or reflect the color of the tab... maybe a coral pink for inputs?
plan what changes will be made and how you will do it and ask me for any clarifications 








Consider the security and privacy of `sd serve` mode
- are api-keys stored anywhere when users submit them? they must NOT be stored anywhre on the server, not in logs, or in config etc. can we give guarantees ont his?
- what happens with uploaded files? can they also be ephemeral? can we make sure they are not persisted to disk. using tmp files is OK, provided they get cleaned up, but in memory might be better?
- anything else we shoudl consider? users might use it for inappropriate files...
- can we also allow users to specify API_BASE as well as API_KEY? it's ok to use LLM_API_BASE as a default if it's present. if user doesn't specify and we can't find one then make an error when user tries to execute a prompt

anything else to consider?






Can we add a 'cheatsheet' to the UI? I would like this implemented as a popover with key concepts and examples of how to use struckdown syntax
it should also show keybaord shortcuts
can we implement shortcuts for 
- inputs (ctrl + i)
- settings (ctrl + ,)
- save (ctrl + s or ctrl + enter)

the popover should have links to the documentation for more advanced usage
make a plan.

