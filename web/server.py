import ast
import numpy as np
import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq_cli.generate import get_symbols_to_strip_from_output


class NaiveTranslator():
    def __init__(self, cfg):
        self.cfg = cfg

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


def cli_main():
    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser)
    translator = NaiveTranslator(convert_namespace_to_omegaconf(args))

    # start server ...
    # when a post request arrives, just call `translator.translate(input)` to get the translation and return the result.
    # for example, call `translator.translate("这个翻译系统怎么样？")`, you will get:
    # > Source: 这个翻译系统怎么样？
    # > Target: What about this translation system?

if __name__ == "__main__":
    cli_main()
