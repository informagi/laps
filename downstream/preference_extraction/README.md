## Models
The models can be downloaded from the following links:
- [Fine-tuned FlanT5 Large model (Recipe)](http://gem.cs.ru.nl/grad-pkg/flan-t5-finetuned-model-large.tar.gz)
- [Fine-tuned FlanT5 Large model (Movie)](http://gem.cs.ru.nl/grad-pkg/flan-t5-further-finetuned-model-large-movie.tar.gz)
  - This model is first fine-tuned on the recipe domain and then further fine-tuned on the movie domain.

**How to use the models:**
These models are Huggingface transformers models and can be used in the same way as other Huggingface models. 
See [here](https://huggingface.co/docs/transformers/en/model_doc/flan-t5) for more information on how to use these models.


## Prompt
`prompts.py` contains the prompts used for the recipe and movie domain.
Each preference has a corresponding prompt, which asks the model to extract the value of that preference from the given dialogue.
