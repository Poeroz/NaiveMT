import os
import sys
import ast
import numpy as np
import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq_cli.generate import get_symbols_to_strip_from_output

fairseq_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class NaiveTranslator():
    def __init__(self):
        # set config
        parser = options.get_interactive_generation_parser()
        config_dir = os.path.join(fairseq_dir, "wmt_zhen")
        input_args = [config_dir]
        args = options.parse_args_and_arch(parser, input_args)
        args.source_lang = "zh"
        args.target_lang = "en"
        args.beam = 4
        args.path = os.path.join(config_dir, "model.pt")
        args.tokenizer = "moses"
        args.bpe = "subword_nmt"
        args.bpe_codes = os.path.join(config_dir, "bpecodes")
        self.cfg = convert_namespace_to_omegaconf(args)

        # set batch size
        self.cfg.dataset.batch_size = 1

        # fix seed for stochastic decoding 
        np.random.seed(self.cfg.common.seed)
        utils.set_torch_seed(self.cfg.common.seed)
        
        # setup task, e.g. translation
        self.task = tasks.setup_task(self.cfg.task)

        # load model
        overrides = ast.literal_eval(self.cfg.common_eval.model_overrides)
        self.models, _model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(self.cfg.common_eval.path),
            arg_overrides=overrides,
            task=self.task,
            suffix=self.cfg.checkpoint.checkpoint_suffix,
            strict=(self.cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=self.cfg.checkpoint.checkpoint_shard_count,
        )

        # set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        # optimize ensemble for generation
        for model in self.models:
            if model is None:
                continue
            model.cuda()
            model.prepare_for_inference_(self.cfg)
        
        # initialize generator
        self.generator = self.task.build_generator(self.models, self.cfg.generation)

        # tokenization and BPE
        self.tokenizer = self.task.build_tokenizer(self.cfg.tokenizer)
        self.bpe = self.task.build_bpe(self.cfg.bpe)

        # Load alignment dictionary for unknown word replacement
        self.align_dict = utils.load_align_dict(self.cfg.generation.replace_unk)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in self.models]
        )

    def encode_fn(self, x):
        if self.tokenizer is not None:
            x = self.tokenizer.encode(x)
        if self.bpe is not None:
            x = self.bpe.encode(x)
        return x

    def decode_fn(self, x):
        if self.bpe is not None:
            x = self.bpe.decode(x)
        if self.tokenizer is not None:
            x = self.tokenizer.decode(x)
        return x
    
    def get_tokens_and_lengths(self, input):
        tokens = self.task.source_dictionary.encode_line(
            self.encode_fn(input), add_if_not_exist=False
        ).long().unsqueeze(0).cuda()
        lengths = torch.tensor([t.numel() for t in tokens]).cuda()
        return tokens, lengths

    def translate(self, input):
        src_tokens, src_lengths = self.get_tokens_and_lengths(input)
        sample = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
            },
        }
        translations = self.task.inference_step(
            self.generator, self.models, sample,
        )
        src_tokens = utils.strip_pad(src_tokens[0], self.tgt_dict.pad())
        hypos = translations[0]
        src_str = self.src_dict.string(src_tokens, self.cfg.common_eval.post_process)
        hypo = hypos[0] # top 1 translation
        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
            hypo_tokens=hypo["tokens"].int().cpu(),
            src_str=src_str,
            alignment=hypo["alignment"],
            align_dict=self.align_dict,
            tgt_dict=self.tgt_dict,
            remove_bpe=self.cfg.common_eval.post_process,
            extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.generator),
        )
        detok_hypo_str = self.decode_fn(hypo_str)

        print("Source: {}".format(input))
        print("Target: {}".format(detok_hypo_str))

        return detok_hypo_str


if __name__ == "__main__":
    translator = NaiveTranslator()
    translator.translate("??????????????????????????????")