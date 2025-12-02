when I do this, I would prefer it to happen in a single step:

curl https://www.amazon.co.uk/Apple-Mac-mini-Computer-M4/dp/B0DLBSGKWV > x.html
uv run sd chat "{{source}} Extract the product data [[product:x]]" -s x.html

So I would like to do this:

uv run sd chat "{{source}} Extract the product data [[product:x]]" -s https://www.amazon.co.uk/Apple-Mac-mini-Computer-M4/dp/B0DLBSGKWV

And have SD recognise that it is a URL and fetch it and extract the data.

Web pages can be very long though and waste tokens so we should pre-process

from readability import Document
import markdownify
import requests

url = "https://example.com"
html = requests.get(url).text
doc = Document(html)
main = doc.summary()         # cleaned, article-only HTML
md = markdownify.markdownify(main, heading_style="ATX")
print(md)



make a plan to combine -s URL with readability and then pass as source.

have a --raw flag that will just pass the URL as source
the default will be to parse the html

add an example of use to the README



At the same time, add a struckdown an action which allows me to 
- fetch contents of a URL and 
- extract data from that  URL in same way (readability, markdown, etc)


So, in a sd prompt I would 

```

[[@fetch:website_content|<url_from_input_data>]]

-------

extract the product data from this page
[[product:myproduct]]

```


And this would take `url_from_input_data` from an input xlsx file and fetch the contents of the URL and then extract the product data from the page.


So =

- add the struckdown action
- use requests library to fetch content
- write code to parse contents of fetched url using readability
- the action dumps the markdown content into the context

Then

- re-use the functions above to allow passing a url to -s when doing `sd chat` and have it auto-fetched and parsed


make a plan and ask about how to handle edge cases you identify








