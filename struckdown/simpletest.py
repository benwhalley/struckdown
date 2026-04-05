from struckdown import complete

xx = """pick the green one [[pick:fruit|{{cc}}]]"""

complete(xx, context={"cc": "apple,orange,banana"}).response
