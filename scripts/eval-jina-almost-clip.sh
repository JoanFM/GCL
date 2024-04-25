data_root="./data"
eval_root="./eval"
#model_ckpt="/path/model.pth"

#rm -rf $eval_root/VITB32-Pretrain

mkdir $eval_root/jina-almost-clip

python evals/eval_gs_v1.py \
      --model_name jina-almost-clip \
      --test_csv $data_root/marqo-gs-dataset/marqo_gs_wfash_1m/query_0_product_id_0.csv \
      --doc-meta $data_root/marqo-gs-dataset/marqo_gs_wfash_1m/corpus_1.json \
      --weight_key "score_linear" \
      --output-dir $eval_root/jina-almost-clip/eval_train_e20 \
      --batch-size 10 \
      --num_workers 4 \
      --left-key "['query']" \
      --right-key "['image_local']" \
      --img-or-txt "[['txt'], ['img']]" \
      --left-weight "[1]" \
      --right-weight "[1]" \
      --context-length "[[77], [0]]" \
      --top-q 2000


python evals/eval_gs_v1.py \
      --model_name jina-almost-clip \
      --test_csv $data_root/marqo-gs-dataset/marqo_gs_wfash_1m/query_0_product_id_1.csv \
      --doc-meta $data_root/marqo-gs-dataset/marqo_gs_wfash_1m/corpus_2.json \
      --weight_key "score_linear" \
      --output-dir $eval_root/jina-almost-clip/eval_q0d1_e20 \
      --batch-size 10 \
      --num_workers 4 \
      --left-key "['query']" \
      --right-key "['image_local']" \
      --img-or-txt "[['txt'], ['img']]" \
      --left-weight "[1]" \
      --right-weight "[1]" \
      --context-length "[[77], [0]]" \
      --top-q 2000


python evals/eval_gs_v1.py \
      --model_name jina-almost-clip \
      --test_csv $data_root/marqo-gs-dataset/marqo_gs_wfash_1m/query_1_product_id_0.csv \
      --doc-meta $data_root/marqo-gs-dataset/marqo_gs_wfash_1m/corpus_1.json \
      --weight_key "score_linear" \
      --output-dir $eval_root/jina-almost-clip/eval_q1d0_e20 \
      --batch-size 10 \
      --num_workers 4 \
      --left-key "['query']" \
      --right-key "['image_local']" \
      --img-or-txt "[['txt'], ['img']]" \
      --left-weight "[1]" \
      --right-weight "[1]" \
      --context-length "[[77], [0]]" \
      --top-q 2000


python evals/eval_gs_v1.py \
      --model_name jina-almost-clip \
      --test_csv $data_root/marqo-gs-dataset/marqo_gs_wfash_1m/query_1_product_id_1.csv \
      --doc-meta $data_root/marqo-gs-dataset/marqo_gs_wfash_1m/corpus_2.json \
      --weight_key "score_linear" \
      --output-dir $eval_root/jina-almost-clip/eval_q1d1_e20 \
      --batch-size 10 \
      --num_workers 4 \
      --left-key "['query']" \
      --right-key "['image_local']" \
      --img-or-txt "[['txt'], ['img']]" \
      --left-weight "[1]" \
      --right-weight "[1]" \
      --context-length "[[77], [0]]" \
      --top-q 2000


python evals/aggregate_results.py \
      --exp-path $eval_root/jina-almost-clip \
      --evals "['eval_train_e20', 'eval_q0d1_e20', 'eval_q1d0_e20', 'eval_q1d1_e20']"

