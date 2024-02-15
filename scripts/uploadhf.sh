MODELNAMES=(
    "GritLM-7B"
    "GritLM-8x7B"
)

for MODELNAME in "${MODELNAMES[@]}"; do
    huggingface-cli upload --private GritLM/$MODELNAME ./$MODELNAME
done

