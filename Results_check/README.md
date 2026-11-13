# Analysis entry points for current deduplicated results

The analysis notebooks in this directory use the implementation in
`../PPHF_modules/results_check/`. The GitHub snapshot includes result PDFs and
source scripts, but omits result CSVs and saved notebook outputs.
Run the complete ordinary FHF/HFF/HFFPS experiments for both datasets before the
four analysis notebooks. They read newly generated results from `fhf_dedup_*`
and create outputs in `deduplication_results/`.
