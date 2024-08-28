
import click
# Can be run with python -m pie.scripts.group
import pie.utils


@click.group()
def pie_cli():
    """ Group command for Pie """


@pie_cli.command()
@click.option('--model_spec', help="Path to model(s)")
@click.option('--batch_size', type=int, default=50, help="Size of the batch")
@click.option('--device', default='cpu', help="Device to use to run the network")
def webapp(model_spec, batch_size, device):
    """ Run the webapp """
    # Until further version, we should explain what's going on
    print("Not supported anymore, do pip install flask_pie")
    raise Exception("The web version of pie has moved to github.com/"
                    "hipster-philology/flask_pie")


@pie_cli.command()
@click.argument('input_path')
@click.argument('model_spec', type=pie.utils.model_spec)
@click.option('--keep_boundaries', is_flag=True,
              help='Keep boundaries from the original input file')
@click.option('--batch_size', type=int, default=50)
@click.option('--device', default='cpu')
@click.option('--use_beam', is_flag=True, default=False)
@click.option('--beam_width', default=10, type=int)
@click.option('--lower', is_flag=True, help="Treat the input as lower case")
@click.option('--max_sent_len', type=int, default=35,
              help='Split sentences longer than this amount')
@click.option('--vrt', is_flag=True, help='Verticalized input format')
def tag(model_spec, input_path, keep_boundaries, batch_size, device,
        use_beam, beam_width, lower, max_sent_len, vrt):
    """ Tag [INPUT_PATH] with model(s) at [MODEL_SPEC]"""
    import pie.scripts.tag
    pie.scripts.tag.run(
        model_spec, input_path, beam_width, use_beam, keep_boundaries,
        device=device, batch_size=batch_size, lower=lower,
        max_sent_len=max_sent_len, vrt=vrt)


@pie_cli.command("tag-pipe")
@click.argument('model_spec', type=pie.utils.model_spec)
@click.option('--batch_size', type=int, default=50)
@click.option('--device', default='cpu')
@click.option('--use_beam', is_flag=True, default=False)
@click.option('--beam_width', default=10, type=int)
@click.option('--lower', is_flag=True, help="Lowercase input to tagger", default=False)
@click.option('--tokenize', is_flag=True, help="Tokenize the input", default=False)
def tag_pipe(model_spec, device, batch_size, lower, beam_width, use_beam, tokenize):
    """ Tag the terminal input with [MODEL_SPEC]"""
    import pie.scripts.tag_pipe
    pie.scripts.tag_pipe.run(
        model_spec=model_spec, device=device, batch_size=batch_size,
        lower=lower, beam_width=beam_width, use_beam=use_beam, tokenize=tokenize)


@pie_cli.command("eval")
@click.argument('model_path')
@click.argument('test_path', nargs=-1)
@click.option('--train_path', default=None,
              help="File used to compute unknown tokens/targets")
@click.option('--settings', help="Settings file used for training")
@click.option('--batch_size', type=int, default=500)
@click.option('--buffer_size', type=int, default=100000)
@click.option('--device', default='cpu')
@click.option('--model_info', is_flag=True, default=False)
@click.option('--full', is_flag=True, default=False)
@click.option('--confusion', default=False, is_flag=True)
@click.option('--report', default=False, is_flag=True)
@click.option('--markdown', default=False, is_flag=True)
@click.option('--use_beam', is_flag=True, default=False)
@click.option('--beam_width', type=int, default=12)
@click.option('--confusion', default=False, is_flag=True,
              help="Show the confusion matrix for most common terms at least")
@click.option('--report', default=False, is_flag=True, help="Show metrics for each label on top of the class results")
@click.option('--markdown', default=False, is_flag=True, help="Display results in markdown")
@click.option("--export", default=False, is_flag=True, help="Export the data from the evaluation")
@click.option("--export-name", default="full_report.json", help="Name of the exported file")
def evaluate(model_path, test_path, train_path, settings, batch_size,
             buffer_size, device, model_info, full, confusion, report,
             markdown, export, export_name,
             use_beam, beam_width):
    """ Evaluate [MODEL_PATH] against [TEST_PATH] using [TRAIN_PATH] to compute
    unknown tokens"""
    import pie.scripts.evaluate
    pie.scripts.evaluate.run(
        model_path=model_path, test_path=test_path, train_path=train_path,
        settings=settings, batch_size=batch_size, buffer_size=buffer_size,
        device=device, model_info=model_info, full=full, confusion=confusion,
        report=report, markdown=markdown,
        use_beam=use_beam, beam_width=beam_width,
        export_scorer=export, export_name=export_name
    )


@pie_cli.command("train")
@click.argument('config_path')
@click.option('--seed', type=int, default=None)
def train(config_path, seed):
    """ Train a model using the file at [CONFIG_PATH]"""
    import pie.scripts.train
    import pie.settings
    pie.scripts.train.run(pie.settings.settings_from_file(config_path), seed)


@pie_cli.command("info")
@click.argument("model_file", type=click.Path(exists=True,
                                              file_okay=True,
                                              dir_okay=False,
                                              readable=True))
def info(model_file):
    from pie.models import BaseModel
    import pprint
    m = BaseModel.load(model_file)
    bar = "=====================\n"
    click.echo(bar+"Settings", color="red")
    pprint.pprint(m._settings)
    click.echo(bar+"Architecture", color="red")
    click.echo(repr(m))


@pie_cli.command("finetune")
@click.argument('input_model', type=click.Path(exists=True, file_okay=True, readable=True, dir_okay=False))
@click.argument('train', type=click.Path(exists=True, file_okay=True, readable=True, dir_okay=False))
@click.argument('dev', type=click.Path(exists=True, file_okay=True, readable=True, dir_okay=False))
# @click.option('--test', type=str, default=None)
@click.option('--seed', type=int, default=None)
@click.option("--batch-size", type=int, default=None)
@click.option("--epochs", type=int, default=100)
@click.option("--mode", type=click.Choice(["expand", "skip", "replace"]), default="expand")
@click.option("--device", type=str, default="cpu")
@click.option("--out", type=str, default="model")
@click.option("--path", type=str, default="./models/")
@click.option("--lr", type=float, default=None)
def finetune(
        input_model: click.Path, train: click.Path, dev: click.Path,
        seed: int, batch_size: int, device: str, epochs: int, out: str, path: str,
        mode: str, lr: float
):
    from pie.models import BaseModel

    m = BaseModel.load(input_model)
    settings = m._settings #  copy.copy(m._settings)
    settings["input_path"] = train
    settings["dev_path"] = dev
    settings["batch_size"] = batch_size or settings["batch_size"]
    settings["device"] = device or settings["device"]
    settings["epochs"] = epochs or settings["epochs"]
    settings["lr"] = lr or settings["lr"]
    settings["modelname"] = out
    settings["modelpath"] = path
    settings["load_pretrained_model"] = {
        "pretrained": str(input_model),
        "labels_mode": mode
    }

    def upgrade_settings(settings):
        if "lr_scheduler_params" not in settings:
            settings["lr_scheduler_params"] = {}
        if "lr_T_max" in settings:
            settings.lr_scheduler_params["T_max"] = settings["lr_T_max"]

        settings.pretrain_embeddings = False
        settings.load_pretrained_embeddings = ""
        settings.load_pretrained_encoder = ""
        settings["checks_per_epoch"] = 1
        # settings["report_freq"] = None

    upgrade_settings(settings)

    import pie.scripts.train
    import pie.settings
    pie.scripts.train.run(settings, seed=seed or 42)


if __name__ == "__main__":
    pie_cli()
