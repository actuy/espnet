#!/usr/bin/env bash
#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0       # start from -1 if you need to start from data download
stop_stage=100
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train.yaml # current default recipe requires 4 gpus.
                             # if you do not have 4 gpus, please reconfigure the `batch-bins` and `accum-grad` parameters in config.
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=0               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/blob/v-chzh/dataDir/data/libri
dataprefix=${datadir}/LibriSpeech/espnet
#dataprefix=.
#dataprefix=${datadir}/espnet
dumpdir=${dataprefix}/${dumpdir}
#datadir=/export/a15/vpanayotov/data

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
recog_set="test_clean test_other dev_clean dev_other"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/LibriSpeech/${part} ${dataprefix}/data/${part//-/_}
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=${dataprefix}/fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in train_other_500; do
#    for x in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            ${dataprefix}/data/${x} ${dataprefix}/exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh ${dataprefix}/data/${x}
    done

    utils/combine_data.sh --extra_files utt2num_frames ${dataprefix}/data/${train_dev}_org ${dataprefix}/data/dev_clean ${dataprefix}/data/dev_other
    utils/combine_data.sh --extra_files utt2num_frames ${dataprefix}/data/${train_set}_org ${dataprefix}/data/train_clean_100 ${dataprefix}/data/train_clean_360 ${dataprefix}/data/train_other_500

     remove utt having more than 3000 frames
     remove utt having more than 400 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 ${dataprefix}/data/${train_set}_org ${dataprefix}/data/${train_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 ${dataprefix}/data/${train_dev}_org ${dataprefix}/data/${train_dev}

    # compute global CMVN
    compute-cmvn-stats scp:${dataprefix}/data/${train_set}/feats.scp ${dataprefix}/data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    echo "Info: doing split of tr dir"
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/librispeech/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    echo "Info: doing split of dt dir"
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/librispeech/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        ${dataprefix}/data/${train_set}/feats.scp ${dataprefix}/data/${train_set}/cmvn.ark ${dataprefix}/exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        ${dataprefix}/data/${train_dev}/feats.scp ${dataprefix}/data/${train_set}/cmvn.ark ${dataprefix}/exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            ${dataprefix}/data/${rtask}/feats.scp ${dataprefix}/data/${train_set}/cmvn.ark ${dataprefix}/exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=${dataprefix}/data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=${dataprefix}/data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p ${dataprefix}/data/lang_char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -f 2- -d" " ${dataprefix}/data/${train_set}/text > ${dataprefix}/data/lang_char/input.txt
    spm_train --input=${dataprefix}/data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece < ${dataprefix}/data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
        ${dataprefix}/data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
        ${dataprefix}/data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            ${dataprefix}/data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
fi