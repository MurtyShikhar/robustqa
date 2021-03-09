import nltk
from collections import Counter

# list of tags: https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/
def count_tags(context, tags_to_find):
    tokens = nltk.word_tokenize(context.lower())
    text = nltk.Text(tokens)
    tags = nltk.pos_tag(text)
    tag_counts = Counter(tag for word, tag in tags)

    count = 0
    for tag in tags_to_find:
        count += tag_counts[tag] if tag in tag_counts else 0
    return count
