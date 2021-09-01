# coding=utf-8

import os
from tqdm import tqdm

import datasets

logger = datasets.logging.get_logger(__name__)

class DocBankLAConfig(datasets.BuilderConfig):
    """BuilderConfig for DocBankLA"""

    def __init__(self, **kwargs):
        """BuilderConfig for DocBankLA.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DocBankLAConfig, self).__init__(**kwargs)

class DocBankLADataset(datasets.GeneratorBasedBuilder):
    """ Subset of the DocBank dataset for document layout analysis. """
    
    BUILDER_CONFIGS = [
        DocBankLAConfig(name="docbank-la", description="DocBank-LA dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "abstract",
                                "author",
                                "caption",
                                "date",
                                "equation",
                                "figure",
                                "footer",
                                "list",
                                "paragraph",
                                "reference",
                                "section",
                                "table",
                                "title",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
        )


    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.manual_dir
        data_dir_basename = os.path.basename(os.path.normpath(data_dir))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={
                    "data_path": f"{data_dir}/{data_dir_basename}_train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={
                    "data_path": f"{data_dir}/{data_dir_basename}_dev"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={
                    "data_path": f"{data_dir}/{data_dir_basename}_test"
                }
            ),
        ]

    def _generate_examples(self, data_path):
        logger.info(f"⏳ Generating examples from {data_path}")
        filenames = sorted(os.listdir(data_path))

        for guid, fname in tqdm(enumerate(filenames), desc=f"Reading files in {data_path}"):
            filepath = os.path.join(data_path, fname)
            words  = []
            bboxes = []
            labels = []

            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    splits = line.split("\t")
                    assert len(splits) == 10
                    word = splits[0]
                    bbox = splits[1:5]
                    label = splits[-1].rstrip()
                    bbox = [int(b) for b in bbox]

                    if word == "##LTFigure##" or word == "##LTLine##":
                        continue

                    words.append(word)
                    bboxes.append(bbox)
                    labels.append(label)

            assert len(words) == len(bboxes)
            assert len(bboxes) == len(labels)

            yield guid, {"id": str(guid), "words": words, "bboxes": bboxes, "tags": labels}
