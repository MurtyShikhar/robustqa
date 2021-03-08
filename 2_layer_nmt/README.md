# 2 layer NMT 

## Language
Source Language: English 

Pivot Languages: Spanish (TED Talks from 2020 CS224N Assignment 4), Vietnamese (IWSLT 2015)

### Vocab Size
English: 16K

Spanish:13K

Vietnamese: 8K

## BackTranslation  

_Use English - Spanish - English with beam size = 1 as an example_

Build the vocab: `sh run_en_es_beam_1.sh vocab`

Train the NMT model that translates English to Spanish: `sh run_en_es_beam_1.sh train`

Train the NMT model that translates Spanish to English: `sh run_es_en_beam_1.sh train`

Translate source English contexts to Spanish contexts: `sh run_en_es_beam_1.sh translate_context`

Translate source English queries to Spanish queries: `sh run_en_es_beam_1.sh translate_queries`

After dropping the non-translatable contexts and queries, we are ready to generate the back-translated English contexts and queries: 

Back-translate Spanish contexts to English contexts: `sh run_es_en_beam_1.sh translate_context`

Back-translate Spanish queries to English queries: `sh run_es_en_beam_1.sh translate_queries`
