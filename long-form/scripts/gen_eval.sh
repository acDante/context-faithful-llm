#!/bin/bash

./scripts/eval_scrolls.sh
cd ../evaluate
./scripts/compute_llmscore_ss.sh
cd ../long-form