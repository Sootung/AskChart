## Data preparation

- It support online and offline ocr. For offline ocr, you should first download [eval.zip](https://drive.google.com/file/d/14h1G_k0KyeizBoYo6LqOe_EQnqapifDU/view?usp=sharing). For online ocr, you should download the test images and annotation file from the corresponding benchmark, and set the argument 'online_ocr'.
- It contains custom annotations, scripts, and the prediction files with LLaVA v1.5. Extract to `eval`. This also provides a general structure for all datasets.

After downloading all of them, organize the data as follows in `eval`.

```Shell
eval
├── chartqa
│   ├── answers
│   ├── test
│   ├── instruct_ChartQA_test_augmented_OCR_test.jsonl
|   └── instruct_ChartQA_test_human_OCR_test.jsonl
├── chart2text
│   ├── answers
│   ├── instruct_chart2text_statista_OCR_test.jsonl
│   └── instruct_chart2text_pew_OCR_test.jsonl
├── OpenCQA
│   ├── answers
│   └── instruct_OpenCQA_test.jsonl
└── chart2table
    ├── answers
    ├── instruct_ChartQA_2table_augmented_OCR_test.jsonl
    └── instruct_ChartQA_2table_human_OCR_test.jsonl
```


## Validating
Our image validation code comes from MoE-LLaVA, thanks for their contribution! 

You can refer to the official repository for validation, but we also provide [off-the-shelf](scripts/v1/eval) scripts.


### Chartqa

1. Download [`chartqa_test`](https://github.com/vis-nlp/ChartQA) and put it under `eval/chartqa`.
2. Single-GPU inference.

**MoE-based** model
```Shell
bash scripts/v1/eval/chartqa.sh
```

### Chart2text

1. Download [`Chart2text`](https://github.com/vis-nlp/Chart-to-text) and put under `askchart_data`.
2. Single-GPU inference.

**MoE-based** model
```Shell
bash scripts/v1/eval/chart2text.sh
```

### OpenCQA

1. Download [`OpenCQA`](https://github.com/vis-nlp/OpenCQA) and put under `askchart_data`.
2. Single-GPU inference.

**MoE-based** model
```Shell
bash scripts/v1/eval/OpenCQA.sh
```

### chart2table

1. Download [`chartqa_test`](https://github.com/vis-nlp/ChartQA) and put under `eval/chartqa`.
2. Single-GPU inference.

**MoE-based** model
```Shell
bash scripts/v1/eval/chart2table.sh
```
