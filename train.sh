#!/usr/bin/env bash
curtime()
{
    date '+%Y-%m-%d %H:%M:%S'
}
dir=log/$(curtime)
echo logs saving to ${dir}
mkdir -p "${dir}"
nohup python -u toutiao_qa_eval.py "$(curtime)" >"${dir}/train.log" 2>&1 &
sleep 1
tail -f "${dir}/train.log"