CONV="phi"
CKPT_NAME="askchartphi-2.7b-finetune-moe"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="askchart_data/eval"
deepspeed askchart/eval/model_loader.py \
    --model-path ${CKPT} \
    --image_tower checkpoints/askchartphi-2.7b-finetune-moe/image_tower \
    --question-file ${EVAL}/chartqa/instruct_ChartQA_test_augmented_OCR.jsonl \
    --image-folder ${EVAL}/chartqa/test/png \
    --answers-file ${EVAL}/chartqa/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --top_p 2 \
    --conv-mode ${CONV}

python3 -m askchart.eval.eval_chartqa \
    --annotation-file ${EVAL}/chartqa/instruct_ChartQA_test_augmented_OCR.jsonl \
    --result-file ${EVAL}/chartqa/answers/${CKPT_NAME}.jsonl