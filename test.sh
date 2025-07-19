export OPENAI_API_KEY="sk-Rlk4nftf04HJqggDupZ7uM4Ur7TNUIHhlAlStDI2hCQtTLc5"
export OPENAI_API_BASE="https://api.chatanywhere.tech/v1"

cd src
python test.py --llm_model_name gpt-4 \
    --output_dir ../datasets/R2R/exprs/gpt-4-val-unseen \
    --val_env_name R2R_val_instr \