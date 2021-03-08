# 2-Layer NMT 

## Language
- Source Language: English 

- Pivot Languages: Spanish (TED Talks from 2020 CS224N Assignment 4), Vietnamese (IWSLT 2015) 

*Need to find the exact source of the Spanish Language*

### Vocab Size

- **English: 16K**

- **Spanish:13K**

  - Spanish vocab size is determined by domain knowledge in linguistics. 

- **Vietnamese: 8K**

  - Vietnamese Vocab size is determined according to the benchmark listed here: [Stanford NLP (word-level)](https://nlp.stanford.edu/projects/nmt/), [Tensorflow NMT (word level)](https://github.com/tensorflow/nmt#iwslt-english-vietnamese)

## Hyperparameter

- 2 hidden layer (benchmark: [Tensorflow NMT](https://github.com/tensorflow/nmt#iwslt-english-vietnamese), [Stanford NLP](https://nlp.stanford.edu/projects/nmt/))

- learning rate: 7.5e-4 (by experiment)

- patience = 2 (by experiment)

- batch size = 64 (by experiment) 

- dropout = 0.2 (benchmark: [Tensorflow NMT](https://github.com/tensorflow/nmt#iwslt-english-vietnamese), [Stanford NLP (word-level)](https://nlp.stanford.edu/projects/nmt/))

- hidden size = 512 (benchmark: [Tensorflow NMT](https://github.com/tensorflow/nmt#iwslt-english-vietnamese))

- embed size = 512 (benchmark: [Tensorflow NMT](https://github.com/tensorflow/nmt#iwslt-english-vietnamese))

## Back-Translation  

_Use English - Spanish - English with beam size = 1 as an example_

- Build the vocab: `sh run_en_es_beam_1.sh vocab`

- Train the NMT model that translates English to Spanish: `sh run_en_es_beam_1.sh train`

- Train the NMT model that translates Spanish to English: `sh run_es_en_beam_1.sh train`

- Translate source English contexts to Spanish contexts: `sh run_en_es_beam_1.sh translate_context`

- Translate source English queries to Spanish queries: `sh run_en_es_beam_1.sh translate_queries`

After dropping the non-translatable contexts and queries, we are ready to generate the back-translated English contexts and queries: 

- Back-translate Spanish contexts to English contexts: `sh run_es_en_beam_1.sh translate_context`

- Back-translate Spanish queries to English queries: `sh run_es_en_beam_1.sh translate_queries`
