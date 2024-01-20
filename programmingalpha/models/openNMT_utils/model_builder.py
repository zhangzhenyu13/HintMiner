from onmt.utils.logging import logger
from programmingalpha.models.TextGenModels import TextGeneratorModel
import onmt
import codecs
from ..translate.translator import Translator
from onmt.utils.parse import ArgumentParser
import onmt.inputters as inputters
import torch
from onmt.decoders.ensemble import EnsembleModel
from ..translate.beam import GNMTGlobalScorer
def buildModelForPrediction(model_path=None,checkpoint=None,model_opt=None,fields=None):
    TextGeneratorModel.model_opt=model_opt
    TextGeneratorModel.opt=model_opt
    TextGeneratorModel.fields=fields
    textGen=TextGeneratorModel()
    if model_opt is not None:
        print(model_opt)
    textGen.loadModel(model_path,checkpoint)
    return textGen.transformer

#def build_model(model_opt, opt, fields, checkpoint):
#    logger.info('Building model...')
#    model = buildModelForPrediction(checkpoint)
#    logger.info(model)
#    return model

def load_test_model_one(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocab = checkpoint['vocab']

    if inputters.old_style_vocab(vocab):
        fields = inputters.load_old_vocab(
            vocab, opt.data_type, dynamic_dict=model_opt.copy_attn
        )
    else:
        fields = vocab
    print(opt)
    model = buildModelForPrediction(checkpoint=checkpoint,model_opt=model_opt,fields=fields)

    if opt.fp32:
        model.float()
    model.eval()
    model.generator.eval()


    return fields, model, model_opt

def load_test_model_ensemble(opt):
    """Read in multiple models for ensemble."""
    shared_fields = None
    shared_model_opt = None
    models = []
    for model_path in opt.models:
        fields, model, model_opt = \
            load_test_model_one(opt, model_path=model_path)
        if shared_fields is None:
            shared_fields = fields
        else:
            for key, field in fields.items():
                try:
                    f_iter = iter(field)
                except TypeError:
                    f_iter = [(key, field)]
                for sn, sf in f_iter:
                    if sf is not None and 'vocab' in sf.__dict__:
                        sh_field = shared_fields[key]
                        try:
                            sh_f_iter = iter(sh_field)
                        except TypeError:
                            sh_f_iter = [(key, sh_field)]
                        sh_f_dict = dict(sh_f_iter)
                        assert sf.vocab.stoi == sh_f_dict[sn].vocab.stoi, \
                            "Ensemble models must use the same " \
                            "preprocessed data"
        models.append(model)
        if shared_model_opt is None:
            shared_model_opt = model_opt
    ensemble_model = EnsembleModel(models, opt.avg_raw_probs)
    return shared_fields, ensemble_model, shared_model_opt

def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    load_test_model = load_test_model_ensemble \
        if len(opt.models) > 1 else load_test_model_one
    fields, model, model_opt = load_test_model(opt)

    scorer = GNMTGlobalScorer.from_opt(opt)

    translator = Translator.from_opt(
        model,
        fields,
        opt,
        model_opt,
        global_scorer=scorer,
        out_file=out_file,
        report_score=report_score,
        logger=logger
    )
    return translator
