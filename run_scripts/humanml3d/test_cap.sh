torchrun --nproc_per_node=1 \
    --master_port=34655 \
    train.py \
    --cfg-path lavis/projects/malmm/cap_humanml3d.yaml \
    --options \
    model.arch blip2_vicuna_instruct \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.num_query_token 32 \
    model.vit_precision fp16 \
    model.freeze_vit True \
    model.memory_bank_length 40 \
    model.num_frames 80 \
    run.init_lr 1e-5 \
    run.max_epoch 10 \
    run.num_beams 5 \
    run.batch_size_train 32 \
    run.batch_size_eval 32 \
    run.accum_grad_iters 2 \
    run.num_workers 8 \
    run.seed 42 \
    run.evaluate True \
    run.valid_splits "['val', 'test']" \
    run.report_metric True \
    run.prefix test \
    run.resume_ckpt_path lavis/output/humanml3d_caption/blip2_vicuna_instruct_vicuna7b/train/b48_e10_lr1e-05_wd0.05_q32_f80_fb40_freezevit/checkpoint_latest.pth
    

