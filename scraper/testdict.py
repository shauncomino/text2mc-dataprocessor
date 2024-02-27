import pandas as pd 
import os

file_path = os.path.join(os.path.abspath('.'), 'projects.csv')
data = {'www.uniquelink1.com': {'download_url': 'www.downloadurl1.com', 'descriptor_string': 'gothic cathedral, dark, large'},
          'www.uniquelink2.com': {'download_url': 'www.downloadurl2.com', 'descriptor_string': 'wood cabin, cozy, small'}}

data.update({"www.uniquelink3.com": {'download_url': 'www.downloadurl3.com', 'descriptor_string': 'some more, string, descriptors'} })

df = pd.read_csv("projects.csv")
data_dict = df.set_index('PROJECT_PAGE_LINK').T.to_dict('dict')
print(data_dict)

print(data_dict.get("some new link"))

keys = list(data_dict.keys())
new_df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['PROJECT_PAGE_LINK', 'PROJECT_DOWNLOAD_LINK', 'DESCRIPTORS'])
new_df['PROJECT_PAGE_LINK'] = keys

new_df.to_csv(file_path, index=False)


"""
PROJECT_PAGE_LINK,PROJECT_DOWNLOAD_LINK,DESCRIPTORS
"pagelink1","downloadlink1","descriptors1"
"pagelink2","downloadlink2","descriptors2"
"pagelink3","downloadlink3","descriptors3"
"""