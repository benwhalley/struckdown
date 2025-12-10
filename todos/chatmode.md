i have a new idea for a feature.

we have the --interactive flag, which will run the chat interface in a terminal
it would also be great if we could run the same in a webui

can we create a new feature within the `sd edit` command which woul allow chatting with the prompt, and build up a conversation history so we can use and test the [[@history]] tag?

this feature would need to allow a text entry box like for {{vars}} which if the @history tag is used, would be shown. this would be a simple textarea where each line is interpreted as a new turn in the conversation. the first line would be the user role, the second line would be interepretted as the `assistant` role in reply

ideally when we are chatting with the prompt we would also be able to see the `thinking` completions slots streamed to a separate area

so, the ui would be 2 columns:

- left column: the prompt editor as current
- the right column: split in two panes one above the other. the top pane is the chat interface which has each turn of the chat displayed in blue/grey areas like an iOS chat window. below the messages is a text entry box and an icon for the `send` button. The bottom pane shows the thinking of the current turn (completions are shown incrementally as they arrive). 

in the chat pane, if the user clicks on a previous llm-created message then the `thinking` slots associated with that message would be shown.

as chat history builds up, the server stores a list of chatterresults and maybe also metadata created during the chat. this can be downloaded as a json file within the chat interface pane (a small discreet text link at the top of the chat pane)

ultrathink about the implentation and UI

keep the save button where it is, but move the "run" button to the right
above the results pane - this is for single-response mode

To switch to chat mode, we will have a toggle in the top menu bar. the toggle button will be right aligned and labelled with the word "Chat mode". when chat mode is on, the Current output pane will be replaced with the chat and results panes I described above.

any questions for me?