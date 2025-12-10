I wanted to find a new feature in Struckdown.
We already have a history plugin, which allows us to insert assistant and user messages into the context.
We can also pass a history file in when using SD chat.
I would like to be able to have an interactive mode where the user can actually chat to the prompt and where assistant and user messages are appended to the history which was passed in as a file.

so `sd chat --history history.txt --interactive` would allow the user to chat with the assistant and the assistant to respond to the user. each time the original history would be supplemented. the command would not exit when the prompts were completed, instead we would have a new prompt for the user to enter.

make a plan ultrathink