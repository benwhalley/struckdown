from struckdown import chatter

xx = """pick the green one [[pick:fruit|{{cc}}]]"""

chatter(xx, context={"cc": "apple,orange,banana"}).response
