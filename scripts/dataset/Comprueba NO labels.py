import xml.etree.cElementTree as ET
from PIL import Image

mainpath_ann = "./0 0.xml"
annotation_xml = ET.parse(mainpath_ann)
root = annotation_xml.getroot()

for j in root:
    data_string = j.text

print(data_string)

#if data_string == "- ":


img = Image.open("./45.png").convert("RGB")
print(img.size[1])



