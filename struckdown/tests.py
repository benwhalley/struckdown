from struckdown import chatter

chatter("tell a joke [[joke]]")
chatter("tell a joke? [[joke]]", extra_kwargs={"max_tokens": 5}).response
