# %%
import re
from pathlib import Path

#%%

iter = 0
buffer = ""
with open("ED_data/ED_data.xml") as input:
	for line in input:
		buffer += line

		if "</page>" in line:
			title = re.findall(r"<title>(.*)</title>", buffer)
			id = re.findall(r"<id>(.*)</id>", buffer)

			filename = id[0] + "-" + title[0].replace(" ", "_").replace("/", "__")[:50]
			pagefile = Path("pages/" + filename + ".xml")
			
			if pagefile.exists():
				print("%s already exists... skipping" % filename)
				buffer = ""
				continue

			with pagefile.open("w") as output:
				output.write(buffer)
				buffer = ""
				print("Extracted %s..." % filename)
# %%
