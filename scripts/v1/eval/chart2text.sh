CONV="phi"
CKPT_NAME="askchartphi-2.7b-finetune-moe"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="askchart_data"
deepspeed --master_port $MASTER_PORT askchart/eval/model_loader.py \
    --model-path ${CKPT} \
    --question-file ${EVAL}/eval/chart2text/instruct_chart2text_statista_OCR.jsonl \
    --image-folder ${EVAL} \
    --answers-file ${EVAL}/eval/chart2text/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --top_p 2 \
    --conv-mode ${CONV}

python3 -m askchart.eval.eval_chart2text \
    --annotation-file ${EVAL}/eval/chart2text/instruct_chart2text_statista_OCR.jsonl \
    --result-file ${EVAL}/eval/chart2text/answers/${CKPT_NAME}.jsonl