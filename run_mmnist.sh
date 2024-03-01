epochs=100
learning_rate=0.001
use_gpu=True

source "C:/Users/Ibram Medhat.DESKTOP-F17GSII/anaconda3/scripts/activate" LTSF_Linear

python -u main.py \
    --epochs $epochs \
    --lr $learning_rate \
    --device "cuda"> results/Debug/initial_mmnist_results.log
