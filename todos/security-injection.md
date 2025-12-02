i'm worried about security and the risk that 
- a slot completion produces outout
- this is included in the next phase of the prompt execution
- what if the output includes [[@actions]] or other [[slots]]?
- could this lead to action execution or code injection?

I _think_ the system should escape this properly, but I want to explcitily simulate/mock a situaiton where a slot returns malicious text and make sure it can't get executed or acted on

make a plan ultrathink