import os
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm

import torch
import transformers
from transformers import HfArgumentParser, Trainer

from helpers import MAX_JP_CODEPOINT, DataArguments, prepare_data, \
    ShibaTrainingArguments, get_model_hyperparams
from masking import RandomSpanMaskingDataCollator, RandomMaskingDataCollator
from shiba import ShibaForAutoregressiveLanguageModeling, CodepointTokenizer


def main():
    transformers.logging.set_verbosity_info()
    # parser = HfArgumentParser((DataArguments, ShibaTrainingArguments))

    # data_args, training_args = parser.parse_args_into_dataclasses()
    data_args = DataArguments(data="/content/all_examples.jsonl")
    training_args = ShibaTrainingArguments(
        logging_steps=50,
        max_steps=60000,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        learning_rate=0.0004,
        adam_beta2=0.98,
        adam_epsilon=1e-06,
        dropout=0.1,
        weight_decay=0.01,
        output_dir="pretrained_shiba",
        masking_type="rand_span",
        gradient_accumulation_steps=6,
        per_device_eval_batch_size=4,
        per_device_train_batch_size=4,
        tpu_num_cores=8,
        num_train_epochs=1,
        tpu_metrics_debug=True,
        report_to="tensorboard"
    )


    tokenizer = CodepointTokenizer()
    if training_args.masking_type == 'bpe_span':
        print('BPE based-span masking')
        data_collator = RandomSpanMaskingDataCollator(tokenizer, True)
    elif training_args.masking_type == 'rand_span':
        print('Random span masking')
        data_collator = RandomSpanMaskingDataCollator(tokenizer, False)
    elif training_args.masking_type == 'rand_char':
        print('Random character masking')
        # char range: https://stackoverflow.com/a/30200250/4243650
        # we aren't including half width stuff
        data_collator = RandomMaskingDataCollator(tokenizer, range(3000, MAX_JP_CODEPOINT))
    else:
        raise RuntimeError('Unknown masking type')

    training_args.logging_dir = training_args.output_dir
    training_data, dev_data = prepare_data(data_args)
    model_hyperparams = get_model_hyperparams(training_args)

    model = ShibaForAutoregressiveLanguageModeling(MAX_JP_CODEPOINT, **model_hyperparams)

    device = xm.xla_device()
    # We wrap this 
    model = model.to(device)

    checkpoint_dir = None
    if training_args.resume_from_checkpoint:
        if training_args.load_only_model:
            model.load_state_dict(torch.load(training_args.resume_from_checkpoint))
        else:
            checkpoint_dir = training_args.resume_from_checkpoint
    # os.environ['WANDB_PROJECT'] = 'shiba'
    # os.environ["WANDB_DISABLED"] = "true"

    print(training_args)
    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=training_data,
                      eval_dataset=dev_data,
                      )

    # trainer.train(resume_from_checkpoint=checkpoint_dir)
    trainer.train()

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == '__main__':
    # main()
    xmp.spawn(_mp_fn, args=(), nprocs=8, start_method='fork')
