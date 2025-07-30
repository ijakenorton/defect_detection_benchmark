export suffix=_default
start_dir=$1

extract_results() {
    local name=$1
    echo "${name} results"
    tail -n 8 ${name}${suffix}/threshold_comparison.txt | head -n 4 | sed 's/\(.*\): \([0-9.]*\):.*/\1: \2/' | awk -F: '{printf "%s: %.4f\n", $1, $2}'
}

names=(mvdsc_mixed devign vuldeepecker diversevul reveal icvul draper juliet cvefixes)

for name in "${names[@]}"; do
    extract_results "$name"
done
