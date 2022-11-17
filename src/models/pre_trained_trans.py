from transformers import ElectraModel, BertModel, BertConfig, RobertaModel, AutoModel

def load_transformer(system:str)->'Model':
    """ downloads and returns the relevant pretrained transformer from huggingface """
    if   system == 'bert-base'    : trans_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    elif system == 'bert-rand'    : trans_model = BertModel(BertConfig())
    elif system == 'bert-large'   : trans_model = BertModel.from_pretrained('bert-large-uncased', return_dict=True)
    elif system == 'bert-tiny'    : trans_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    elif system == 'roberta-base' : trans_model = RobertaModel.from_pretrained('roberta-base', return_dict=True)
    elif system == 'electra-base' : trans_model = ElectraModel.from_pretrained('google/electra-base-discriminator',return_dict=True)
    elif system == 'electra-large': trans_model = ElectraModel.from_pretrained('google/electra-large-discriminator', return_dict=True)
    else: raise ValueError(f"invalid transfomer system provided: {system}")
    return trans_model
