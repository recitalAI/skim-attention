# coding=utf-8

import os
from tqdm import tqdm

import datasets

logger = datasets.logging.get_logger(__name__)

class MixConfig(datasets.BuilderConfig):
    """BuilderConfig for MIX"""

    def __init__(self, **kwargs):
        """BuilderConfig for MIX.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MixConfig, self).__init__(**kwargs)

class MixDataset(datasets.GeneratorBasedBuilder):
    """ Dataset comprised of DocBank, RVL-CDIP and PubLayNet """

    BUILDER_CONFIGS = [
        MixConfig(name="mix", description="MIX dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.manual_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={
                    "index_path": f"{data_dir}/indexed_files/train.txt",
                    "data_path": f"{data_dir}/data",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={
                    "index_path": f"{data_dir}/indexed_files/dev.txt",
                    "data_path": f"{data_dir}/data",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={
                    "index_path": f"{data_dir}/indexed_files/test.txt",
                    "data_path": f"{data_dir}/data",
                }
            ),
        ]

    def _generate_examples(self, index_path, data_path):
        logger.info(f"⏳ Generating examples from {data_path} using {index_path}")
        
        with open(index_path) as f_index:
            split_fnames = [line.rstrip() for line in f_index]
                
        for guid, fname in tqdm(enumerate(split_fnames), desc=f"Reading files in {index_path}"):
            filepath = os.path.join(data_path, fname)

            words = []
            bboxes = []

            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    splits = line.split("\t")
                    word = splits[0]
                    bbox = splits[1:5]
                    bbox = [int(b) for b in bbox]

                    if word == "##LTFigure##" or word == "##LTLine##" or word == "":
                        continue

                    words.append(word)
                    bboxes.append(bbox) # bbox is already normalized

            yield guid, {"id": str(guid), "words": words, "bboxes": bboxes}


