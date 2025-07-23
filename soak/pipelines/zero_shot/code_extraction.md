{{persona}}

You will code the transcript independently, without using a pre-existing codebook. This is the initial coding stage, where insights will be drawn directly from the text. The results from this stage will determine the subsequent theme generation process.

{{research_question}}

You are provided text from transcripts of interviews with participants who took part in the study.

In this stage, you will generate codes. Codes are the foundational units of inductive thematic analysis, capturing significant concepts and ideas from the transcripts. Each code includes a concise name, a meaningful description, and representative quotes from the transcripts.

In this task, a 'code' should be related to the desires, needs, and meaningful outcomes for participants. Codes pertain to participants. Codes describe the feeling of the participant themselves.

Your goal is to carefully look through the text and identify all codes discussed by the participant exhaustively. Each code should be very specific and distinct. You will be rewarded {incentive} if you can complete this effectively.

Identify all relevant codes in the text, provide a Name for each code in 8 to 15 words in sentence case.

Write with concise, concrete details and avoid clichés, generalizations.

Give a dense Description of the code in 80 words and direct Quotes from the participant for each code in around 120 words. These quotes can consist of multiple excerpts from the text.

Try to use the language the participants use when generating the names and descriptions.

Use language like an inductive thematic analysis researcher would do when generating the names and descriptions.

Avoid generalized terms and use specific terms from the quotes when generating code name and description.

Format the response as a JSON output keeping names, descriptions and quotes together in the JSON, and keep them together in Codes.
The response should start from Codes as a JSON output

text: {{chunks}}


[[codes:codes]]