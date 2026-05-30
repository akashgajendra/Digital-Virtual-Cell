from vcpi_prediction_contest import load_test_compounds, load_gene_filter

tc = load_test_compounds()
genes = load_gene_filter()

print(tc.shape)      # should be (N_compounds, ~7 columns)
print(len(genes))    # should be ~13k genes
print(tc.head())