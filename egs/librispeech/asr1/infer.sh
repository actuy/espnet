#!/usr/bin/env bash
# cd /var/storage/shared/xxx/v-chzh/
# git clone https://github.com/actuy/espnet.git
# cd espnet/tools
# ln -sf /home/espnet/tools/* ./
# cd ../egs/librispeech/asr1/
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=16
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

#preprocess_config=
preprocess_config=conf/specaug.yaml
train_config=conf/train.yaml # current default recipe requires 4 gpus.
                             # if you do not have 4 gpus, please reconfigure the `batch-bins` and `accum-grad` parameters in config.
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=no_lm     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5                   # the number of ASR models to be averaged
use_valbest_average=false     # if true, the validation `n_average`-best ASR models will be averaged.
#use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=0               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/blob/v-chzh/dataDir/data/libri
dataprefix=/blob/v-chzh/dataDir/data/libri/espnet
dumpdir=${dataprefix}/${dumpdir}

# base url for downloads.
data_url=www.openslr.org/resources/12

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
train_dev=dev
recog_set="test_clean"
#recog_set="test_clean test_other dev_clean dev_other"

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi

expdir=${dataprefix}/exp/${expname}
mkdir -p ${expdir}
dict=${dataprefix}/data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}

echo "Decoding"
if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
    # Average ASR models
    if ${use_valbest_average}; then
        recog_model=model.val${n_average}.avg.best
        opt="--log ${expdir}/results/log"
    else
        recog_model=model.last${n_average}.avg.best
        opt="--log"
    fi
#    average_checkpoints.py \
#        ${opt} \
#        --backend ${backend} \
#        --snapshots ${expdir}/results/snapshot.ep.* \
#        --out ${expdir}/results/${recog_model} \
#        --num ${n_average}
fi
echo "[info] finish avg ckpt"

feat_recog_dir=${dumpdir}/test_clean/delta${do_delta}
#splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json
#echo "[info] finish split json"

decode_dir=decode_test_clean_${recog_model}_$(basename ${decode_config%.*})_${lmtag}_100
function run() {
    part=$[$1+1]
    export CUDA_VISIBLE_DEVICES=$2
    ${decode_cmd} ${expdir}/${decode_dir}/log/decode.${part}.log \
        asr_recog.py \
        --config ${decode_config} \
        --ngpu 1 \
        --backend ${backend} \
        --batchsize 0 \
        --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.${part}.json \
        --result-label ${expdir}/${decode_dir}/data.${part}.json \
        --model ${expdir}/results/${recog_model}  \
        --api v2
}
jobPerGPU=$[${nj}/${ngpu}]
echo "[info] job per gpu is ${jobPerGPU}"
jobPerGPU=${nj}/${ngpu}

#for ((i=0;i<${nj};i+=${ngpu}));
for ((i=4;i<${nj};i+=${ngpu}));
do
    for ((j=0;j<${ngpu};j++));
    do
        echo "run $(($i+$j)) ${j}"
        run $(($i+$j)) ${j} &
    done

    if [[ $(( $[$i+1]%${jobPerGPU} )) -eq 0 ]]; then
        wait
    fi
    echo "[info] decode $[$i+ngpu]/${nj} done"
    score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
done
#score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
echo "[info] decode all done"

