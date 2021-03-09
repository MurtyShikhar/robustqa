import nltk
from collections import Counter

# list of tags: https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/
def count_tags(context, tags):
    nltk.download('averaged_perceptron_tagger')

    tokens = nltk.word_tokenize(context.lower())
    text = nltk.Text(tokens)
    tags = nltk.pos_tag(text)
    tag_counts = Counter(tag for word, tag in tags)

    count = 0
    for tag in tags:
        count += tag_counts[tag] if tag in tag_counts else 0

    return count
