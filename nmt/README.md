## NMT 

A question answering system that is robust to out-of-distribution datasets, utilizing the method of data augmentation via backtranslation To enhance the generality of the QA model, neural machine translation models (source-to-pivot languages and pivot-to-source languages) were deployed at the subword level (BiLSTM encoder + LSTM decoder architecture with multiplicative attention). 
 
Here is an example using English as source language and German as pivot language:  
- Generate Vocabulary: `sh run_en_de.sh vocab`
- Train NMT model that translates English to German: `sh run_en_de.sh train`
- Test the NMT model: `sh run_en_de.sh test`
- Generate the translation: `sh run_en_de.sh translate_queries/translate_context`
