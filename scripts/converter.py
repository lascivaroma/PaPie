"""
Requirements:

regex
colorama
pycases
click
"""
import os

import itertools
from typing import IO, List, Tuple, Dict
import click
import regex
import cases


def roman_number(inp: str) -> int:
    """
    Source: https://stackoverflow.com/questions/19308177/converting-roman-numerals-to-integers-in-python
    Author: https://stackoverflow.com/users/1201737/r366y
    :param num:
    :return:

    >>> roman_number("XXIV")
    24
    """
    roman_numerals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    result = 0
    for i, c in enumerate(inp.upper()):
        if (i + 1) == len(inp) or roman_numerals[c] >= roman_numerals[inp[i + 1].upper()]:
            result += roman_numerals[c]
        else:
            result -= roman_numerals[c]
    click.secho(f"[Debug] Automatic numeral conversion {inp} to {result}", color="blue")
    return result


def uvji(val):
    return val.replace("V", "U").replace("v", "u").replace("J", "I").replace("j", "i")


def get_tasks(morph):
    if morph == "_":
        return {}
    return dict([
        m.split("=")
        for m in morph.split("|")
    ])


# Useful constants

WRONG_CLITICS = {"quis1"}
DOTS_EXCEPT_APOSTROPHES = r".?!\"“”\"«»…\[\]\(\)„“"
TASKS = "form,lemma,Deg,Numb,Person,Mood_Tense_Voice,Case,Gend,Dis,pos".split(",")


def read_input(opened_file: IO) -> Tuple[Dict[str, List], List[str]]:
    """ Reads an input file. Supports [METADATA:___] in order to split a full doc into multiple chunks
    """
    clitics = []
    header = []
    files = {"full": []}
    cur_text = "full"
    previous_anno = None
    anno = None
    for lineno, line in enumerate(opened_file):
        line = line.strip().split("\t")
        if lineno == 0:
            header = line
            continue

        previous_anno = anno
        anno = dict(zip(header, line))
        if anno["lemma"] == "[METADATA]":
            cur_text = anno["form"]
            files[anno["form"]] = []
            continue

        if anno["form"] in DOTS_EXCEPT_APOSTROPHES:
            if files[cur_text][-1] != {}:
                files[cur_text].append({})
            continue

        anno["Dis"] = "_"

        anno.update(get_tasks(anno["morph"]))
        anno["Mood_Tense_Voice"] = "|".join([
            anno.get(part, "_")
            for part in "Mood_Tense_Voice".split("_")
        ]).replace("_|_|_", "_")

        if anno["lemma"].isnumeric():
            if int(anno["lemma"]) > 3:
                anno["lemma"] = anno["form"] = "3"

        if anno["lemma"][-1].isnumeric() and len(anno["lemma"]) > 1:
            anno["lemma"], anno["Dis"] = anno["lemma"][:-1], anno["lemma"][-1]

        if anno["lemma"] == "[ROMAN_NUMBER]":
            anno["lemma"] = anno["form"] = roman_number(anno["form"])
            if anno["lemma"] > 3:
                anno["lemma"] = anno["form"] = "3"
            else:
                anno["lemma"] = anno["form"] = str(anno["form"])

        if anno["lemma"] == "[Greek]":
            continue

        if anno["POS"] == "OUT":
            click.secho(f"[Debug] Token {anno['form']} is annotated [OUT]", color="blue")

        if anno["POS"] == "PUNC":
            if anno["form"] in "?.;!)(][":
                files[cur_text].append(None)
            continue

        if anno["POS"] == "VERaux":
            anno["POS"] = "VER"

        anno["lemma"] = uvji(anno["lemma"])
        anno["form"] = uvji(anno["form"])

        if len(files[cur_text]) and files[cur_text][-1] and files[cur_text][-1] == previous_anno \
                and files[cur_text][-1]["form"] == anno["form"]:
            if anno["lemma"] not in WRONG_CLITICS:
                clitics.append(anno["lemma"])
                files[cur_text][-1]["lemma"] = files[cur_text][-1]["lemma"] + "界" + anno["lemma"]
                continue
        elif len(files[cur_text]) and files[cur_text][-1] and files[cur_text][-1]["form"] == anno["form"][1:-1]:
            clitics.append(anno["lemma"])
            files[cur_text][-1]["lemma"] = files[cur_text][-1]["lemma"] + "界" + anno["lemma"]
            continue

        if "." in anno["form"]:
            click.secho(f"[Debug] Token {anno['form']} is abbreviated", fg="blue")

        if False and anno["POS"].startswith("NOM"):
            anno["POS"] = "NOM"
        anno["pos"] = anno["POS"]

        files[cur_text].append(anno)
    return {text_id: annots for text_id, annots in files.items()}, clitics

@click.command()
@click.argument("file", type=click.File(mode="r"))
@click.argument("output-dir", type=click.Path(file_okay=False, dir_okay=True))
@click.option("--split", type=bool, default=True)
def command(file: IO, output_dir: str, split: bool):
    content, clitics = read_input(file)
    click.echo(f"Found {len(clitics)} clitics in the source file: {', '.join(set(clitics))}")
    n = "\n"
    click.echo(f"Found {len(content)} different texts in the source file: \n    - {(n+'    - ').join(content)}")
    file.close()
    os.makedirs(output_dir, exist_ok=True)
    if split is False:
        # Insert empty values
        c = list(content.values())
        c[::2] = [] * len(c)
        content = {
            "all": list(itertools.chain.from_iterable(c))
        }
    total = 0
    for file in content:
        local_total = 0
        with open(os.path.join(output_dir, cases.to_kebab(file)+".tsv"), "w") as f:
            f.write("\t".join(TASKS).replace("form", "token") + "\n")
            for annot in content[file]:
                if not annot:
                    f.write("\n")
                    continue
                local_total += 1
                f.write("\t".join([annot.get(h, "_") for h in TASKS]) + "\n")
        click.secho(
            f"Found {local_total:,} tokens in "+ os.path.join(output_dir, cases.to_kebab(file)+".tsv"),
            fg="green"
        )
        total += local_total
    if total != local_total:
        click.secho(
            f"Found {local_total:,} tokens in total",
            fg="green"
        )

command()