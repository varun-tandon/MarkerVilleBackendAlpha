import os, glob
from unidecode import unidecode
import xml.etree.cElementTree as ET
n = 0.0
root = ET.Element("root")
os.chdir("/Users/gmachiraju/Desktop/files")
article_set = ET.SubElement(root, "ArticleSet")
for folder in glob.glob("*"):
    os.chdir("/Users/gmachiraju/Desktop/files/" + str(folder))
    for file in glob.glob("*"):
        n += 1.0
        print(str((n/325947.0)*100) + '%')
        with open(file, 'rb') as f:
            article = ET.SubElement(article_set, "Article")
            ET.SubElement(article, "article-id").text = ''.join([i if ord(i) < 128 else ' ' for i in str(f.name)])
            q =  ' '.join([i if ord(i) < 128 else ' ' for i in f.read()])

            try:
                ET.SubElement(article, 'text').text = str(q[0: q.index('==== Refs')])
            except:
                ET.SubElement(article, 'text').text = str(q)

tree = ET.ElementTree(root)
tree.write('output.xml')




