import json
squad2 = "datasets/train-v2.0.json"
squad1 = "datasets/indomain_train/squad"

squad2_data = json.loads(open(squad2).read())
squad1_data = json.loads(open(squad1).read())

# extracting the titles from SQuAD 2.0 as the topics for the same context in SQuAD 1.1. 
# This approach only compares the titles in SQuAD 1.1, which is the beginning of the 
# context in the paragraphs, with the beginning of the contexts in SQuAD 2.0.
# This solution works around different potential issues:
# 1. Special character differences introduced by Windows or Mac between the two SQuAD versions.
# 2. Inconsistent length of the contexts between two versions. 
# 3. Partial last word in the titles from SQuAD 1.1.
for squad1d in squad1_data['data']:
    topic = None
    header = squad1d['title'].replace("'", '').replace("’", '').lower().split()[:-1]
    #print (header)
    len_header = len(header)
    for i in squad2_data['data']:
        title = i['title']
        for j in i['paragraphs']:
            if j['context'].replace("'", '').replace("’", '').lower().split()[:len_header] == header: 
                #print (j['context'])
                topic = title
                break
        if topic is not None:
            break
    if topic is not None:
        squad1d['topic'] = topic
    else:
        print ("no topic fetched for %s" % squad1d['title'])

with open('datasets/squad_plus_topics_2', 'w') as f:
    json.dump(squad1_data, f)
