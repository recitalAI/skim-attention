from transformers import (
    CONFIG_MAPPING, 
    MODEL_FOR_MASKED_LM_MAPPING, 
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, 
    MODEL_NAMES_MAPPING, 
    TOKENIZER_MAPPING,
    BertTokenizer,
    BertTokenizerFast,
    LongformerTokenizer,
    LongformerTokenizerFast,
)
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, BertConverter, RobertaConverter
from transformers.models.auto.modeling_auto import auto_class_factory

from .modeling.skim_attention import (
    SkimformerConfig,
    BertWithSkimEmbedConfig, 
    SkimmingMaskConfig,
    LongSkimformerConfig,
    SkimformerForMaskedLM,
    SkimformerForTokenClassification, 
    BertWithSkimEmbedForTokenClassification,
    SkimmingMaskForTokenClassification,
    LongSkimformerForMaskedLM,
    LongSkimformerForTokenClassification,
)


CONFIG_MAPPING.update([("skimformer", SkimformerConfig)])
CONFIG_MAPPING.update([("bertwithskimembed", BertWithSkimEmbedConfig)])
CONFIG_MAPPING.update([("skimmingmask", SkimmingMaskConfig)])
CONFIG_MAPPING.update([("longskimformer", LongSkimformerConfig)])
MODEL_NAMES_MAPPING.update([("skimformer", "Skimformer")])
MODEL_NAMES_MAPPING.update([("bertwithskimembed", "BertWithSkimEmbed")])
MODEL_NAMES_MAPPING.update([("skimmingmask", "SkimmingMask")])
MODEL_NAMES_MAPPING.update([("longskimformer", "LongSkimformer")])
TOKENIZER_MAPPING.update([(SkimformerConfig, (BertTokenizer, BertTokenizerFast))])
TOKENIZER_MAPPING.update([(BertWithSkimEmbedConfig, (BertTokenizer, BertTokenizerFast))])
TOKENIZER_MAPPING.update([(SkimmingMaskConfig, (BertTokenizer, BertTokenizerFast))])
TOKENIZER_MAPPING.update([(LongSkimformerConfig, (LongformerTokenizer, LongformerTokenizerFast))])
SLOW_TO_FAST_CONVERTERS.update({"SkimformerTokenizer": BertConverter})
SLOW_TO_FAST_CONVERTERS.update({"BertWithSkimEmbedTokenizer": BertConverter})
SLOW_TO_FAST_CONVERTERS.update({"SkimmingMaskTokenizer": BertConverter})
SLOW_TO_FAST_CONVERTERS.update({"LongSkimformerTokenizer": RobertaConverter})
MODEL_FOR_MASKED_LM_MAPPING.update([(SkimformerConfig, SkimformerForMaskedLM)])
MODEL_FOR_MASKED_LM_MAPPING.update([(LongSkimformerConfig, LongSkimformerForMaskedLM)])
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update([(SkimformerConfig, SkimformerForTokenClassification)])
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update([(BertWithSkimEmbedConfig, BertWithSkimEmbedForTokenClassification)])
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update([(SkimmingMaskConfig, SkimmingMaskForTokenClassification)])
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update([(LongSkimformerConfig, LongSkimformerForTokenClassification)])
AutoModelForMaskedLM = auto_class_factory(
    "AutoModelForMaskedLM", MODEL_FOR_MASKED_LM_MAPPING, head_doc="masked lm"
)
AutoModelForTokenClassification = auto_class_factory(
    "AutoModelForTokenClassification", MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, head_doc="token classification"
)