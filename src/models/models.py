from transformers import logging, ElectraForMultipleChoice, RobertaForMultipleChoice

logging.set_verbosity_error()


def MC_transformer(trans_name:str)->'Model':
    """ downloads and returns miltiple choice transformers"""
    if   trans_name=='electra-base' : trans_model = ElectraForMultipleChoice.from_pretrained("google/electra-base-discriminator", return_dict=True)
    elif trans_name=='electra-large': trans_model = ElectraForMultipleChoice.from_pretrained("google/electra-large-discriminator", return_dict=True)
    elif trans_name=='roberta-base':  trans_model = RobertaForMultipleChoice.from_pretrained("roberta-base", return_dict=True)
    elif trans_name=='roberta-large': trans_model = RobertaForMultipleChoice.from_pretrained("roberta-large", return_dict=True)
    else: raise ValueError(f"{trans_name} is an invalid system: no MCRC model found")
    return trans_model
