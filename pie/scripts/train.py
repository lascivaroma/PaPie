
# Can be run with python -m pie.scripts.train
import logging
import time
import os
from datetime import datetime

import pie
from pie.settings import settings_from_file
from pie.trainer import Trainer
from pie import initialization
from pie.data import Dataset, Reader, MultiLabelEncoder
from pie.models import SimpleModel

# set seeds
import random
import numpy
import torch

logger = logging.getLogger(__name__)


def get_targets(settings):
    return [task['name'] for task in settings.tasks if task.get('target')]


def get_fname_infix(settings, epoch=None):
    # fname
    fname = os.path.join(settings.modelpath, settings.modelname)
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    infix = '+'.join(get_targets(settings))
    if epoch:
        infix += f"-{epoch}"
    infix += '-' + timestamp
    return fname, infix


def run(settings, seed=None):
    now = datetime.now()

    # set seed
    if seed is None:
        if settings.seed == "auto":
            seed = now.hour * 10000 + now.minute * 100 + now.second
        else:
            seed = settings.seed
            assert isinstance(seed, int), "Seed should be an integer"
    print("Using seed:", seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if settings.verbose:
        logging.basicConfig(level=logging.INFO, force=True)

    # datasets
    reader = Reader(settings, settings.input_path)
    tasks = reader.check_tasks(expected=None)
    if settings.verbose:
        print("::: Available tasks :::")
        print()
        for task in tasks:
            print("- {}".format(task))
        print()

    # label encoder
    labels_mode = settings.load_pretrained_model.get("labels_mode")
    labels_mode_accepted = ["expand", "replace_fill", "replace", "skip"]
    assert labels_mode in labels_mode_accepted, \
        f"Invalid value for labels_mode ({labels_mode}), accepted values are {labels_mode_accepted}"
    if settings.load_pretrained_model.get("pretrained") and labels_mode != "replace":
        label_encoder = MultiLabelEncoder.load_from_pretrained_model(
            path=settings.load_pretrained_model["pretrained"],
            new_settings=settings,
            tasks=[t["name"] for t in settings.tasks]
        )
        if settings.load_pretrained_model.get("labels_mode") == "expand":
            if settings.verbose:
                print("::: Fitting/Expanding MultiLabelEncoder with data (expand mode) :::")
                print()
            label_encoder.fit_reader(reader, expand_mode=True)
        elif settings.load_pretrained_model.get("labels_mode") == "replace_fill":
            if settings.verbose:
                print(":: Fitting MultiLabelEncoder with data, completing with parent vocab and labels (replace_fill mode) :::")
                print()
            # Fit a new MultiLabelEncoder with the finetuning data
            new_label_encoder = MultiLabelEncoder.from_settings(settings, tasks=tasks)
            new_label_encoder.fit_reader(reader)

            # Update frequencies with parent LabelEncoders
            for le in label_encoder.all_label_encoders:
                # Skip parent model's tasks that are not in the finetuning config
                if le.name not in ["word", "char"] and le.name not in new_label_encoder.tasks:
                    continue
                # Get the new LabelEncoder to update
                if le.name == "word":
                    new_le = new_label_encoder.word
                elif le.name == "char":
                    new_le = new_label_encoder.char
                else:
                    new_le = new_label_encoder.tasks[le.name]
                # Update frequencies of the new LabelEncoder
                new_le.freqs.update(le.freqs)
                # Update known_tokens if it is a char-based LabelEncoder
                assert le.level == new_le.level, \
                    (f"LabelEncoder levels should be the same between the parent "
                     f"and the child models, found {le.level} and {new_le.level}")
                if new_le.level != "token":
                    new_le.known_tokens.update(le.known_tokens)
                # Expand vocab
                new_le.fitted = False
                new_le.expand_vocab()
            
            # Update uppercase vocab entries
            if new_label_encoder.noise_strategies["uppercase"]["apply"]:
                new_label_encoder.word.register_upper()
                new_label_encoder.char.register_upper()

            # Replace parent label_encoder with the child new_label_encoder
            label_encoder = new_label_encoder
        else:  # "skip"
            if settings.verbose:
                print("::: Fitting MultiLabelEncoder with data (unfitted LabelEncoders only) (skip mode) :::")
                print()
            label_encoder.fit_reader(reader, skip_fitted=True)
    else:  # train from scratch or labels_mode== "replace"
        label_encoder = MultiLabelEncoder.from_settings(settings, tasks=tasks)
        if settings.verbose:
            if settings.load_pretrained_model.get("pretrained"):
                print("::: Fitting MultiLabelEncoder with data (replace mode) :::")
            else:
                print("::: Fitting MultiLabelEncoder with data :::")
            print()
        label_encoder.fit_reader(reader)

    if settings.verbose:
        print()
        print("::: Vocabulary :::")
        print()
        types = '{}/{}={:.2f}'.format(*label_encoder.word.get_type_stats())
        tokens = '{}/{}={:.2f}'.format(*label_encoder.word.get_token_stats())
        print("- {:<15} types={:<10} tokens={:<10}".format("word", types, tokens))
        types = '{}/{}={:.2f}'.format(*label_encoder.char.get_type_stats())
        tokens = '{}/{}={:.2f}'.format(*label_encoder.char.get_token_stats())
        print("- {:<15} types={:<10} tokens={:<10}".format("char", types, tokens))
        print()
        print("::: Tasks :::")
        print()
        for task, le in label_encoder.tasks.items():
            print("- {:<15} target={:<6} level={:<6} vocab={:<6}"
                  .format(task, le.target, le.level, len(le)))
        print()

    trainset = Dataset(settings, reader, label_encoder)

    devset = None
    if settings.dev_path:
        devset = Dataset(settings, Reader(settings, settings.dev_path), label_encoder)
    else:
        logger.warning("No devset: cannot monitor/optimize training")

    # model
    model = SimpleModel(
        label_encoder, settings.tasks,
        settings.wemb_dim, settings.cemb_dim, settings.hidden_size,
        settings.num_layers, cell=settings.cell,
        # dropout
        dropout=settings.dropout, word_dropout=settings.word_dropout,
        # word embeddings
        merge_type=settings.merge_type, cemb_type=settings.cemb_type,
        cemb_layers=settings.cemb_layers, custom_cemb_cell=settings.custom_cemb_cell,
        # lm joint loss
        include_lm=settings.include_lm, lm_shared_softmax=settings.lm_shared_softmax,
        # decoder
        scorer=settings.scorer, linear_layers=settings.linear_layers)

    # pretrain(/load pretrained) embeddings
    if model.wemb is not None:
        if settings.pretrain_embeddings:
            raise ValueError("Pretrained Gensim embedding is not supported in PaPie since 0.3.12. "
                             "Check load_pretrained_embeddings as an alternative.")

        elif settings.load_pretrained_embeddings:
            print("Loading pretrained embeddings")
            if not os.path.isfile(settings.load_pretrained_embeddings):
                print("Couldn't find pretrained embeddings in: {}".format(
                    settings.load_pretrained_embeddings))
            initialization.init_pretrained_embeddings(
                settings.load_pretrained_embeddings, label_encoder.word, model.wemb)

    # load weights from a pretrained encoder
    if settings.load_pretrained_encoder:
        model.init_from_encoder(pie.Encoder.load(settings.load_pretrained_encoder))
    
    if settings.load_pretrained_model.get("pretrained"):
        print(f"Loading pretrained model {settings.load_pretrained_model['pretrained']}")
        model.load_state_dict_from_pretrained(
            settings.load_pretrained_model["pretrained"],
            settings.load_pretrained_model.get("exclude", [])
        )

    # freeze embeddings
    if settings.freeze_embeddings:
        model.wemb.weight.requires_grad = False

    model.to(settings.device)

    print("::: Model :::")
    print()
    print(model)
    print()
    print("::: Model parameters :::")
    print()
    trainable = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    total = sum(p.nelement() for p in model.parameters())
    print("{}/{} trainable/total".format(trainable, total))
    print()

    # training
    print("Starting training")

    running_time = time.time()
    trainer = Trainer(settings, model, trainset, reader.get_nsents())
    scores = None
    try:
        scores = trainer.train_epochs(settings.epochs, devset=devset)
    except KeyboardInterrupt:
        print("Stopping training")
    finally:
        model.eval()
    running_time = time.time() - running_time

    # evaluate best model on devset
    if settings.dev_path:
        print()
        print("Evaluating best model on dev set...")
        print()
        model.eval()
        stored_scores = {}
        with torch.no_grad():
            dev_loss = trainer.evaluate(devset)
            print()
            print("::: Dev losses :::")
            print()
            print('\n'.join('{}: {:.4f}'.format(k, v) for k, v in dev_loss.items()))
            print()
            summary = model.evaluate(devset, trainer.dataset)
            for task_name, scorer in summary.items():
                stored_scores[task_name] = scorer.get_scores()
                scorer.print_summary(scores=stored_scores[task_name])

    # evaluate best model on test set
    if settings.test_path:
        print("Evaluating best model on test set")
        try:
            testset = Dataset(settings, Reader(settings, settings.test_path), label_encoder)
            for task in model.evaluate(testset, trainset).values():
                task.print_summary()
        except Exception as E:
            print(E)

    if devset is not None and not settings.run_test:
        scorers = model.evaluate(devset, trainset)
        scores = []
        for task in sorted(scorers):
            scorer = scorers[task]
            result = scorer.get_scores()
            for acc in result:
                scores.append('{}-{}:{:.6f}'.format(
                    acc, task, result[acc]['accuracy']))
                scores.append('{}-{}-support:{}'.format(
                    acc, task, result[acc]['support']))
        path = '{}.results.{}.csv'.format(
            settings.modelname, '-'.join(get_targets(settings)))
        with open(path, 'a') as f:
            _, infix = get_fname_infix(settings)
            line = [infix, str(seed), str(running_time)]
            line += scores
            f.write('{}\n'.format('\t'.join(line)))

    # save model
    if not settings.run_test:
        # Save best model
        fpath, infix = get_fname_infix(settings, epoch="best")
        fpath = model.save(fpath, infix=infix, settings=settings)
        print("Saved best model to: [{}]".format(fpath))
        # Save last model
        if "last_state_dict" in model.__dict__:
            model.load_state_dict(model.last_state_dict)
            fpath, infix = get_fname_infix(settings, epoch="last")
            fpath = model.save(fpath, infix=infix, settings=settings)
            print("Saved last model to: [{}]".format(fpath))

    print("Bye!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', nargs='?', default='config.json')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument('--opt_path', help='Path to optimization file (see opt.json)')
    parser.add_argument('--n_iter', type=int, default=20, help="Number of iterations for the optimization mode")
    args = parser.parse_args()

    settings = settings_from_file(args.config_path)

    from pie import optimize

    if args.opt_path:
        opt = optimize.read_opt(args.opt_path)
        optimize.run_optimize(run, settings, opt, args.n_iter)
    else:
        run(settings, args.seed)
