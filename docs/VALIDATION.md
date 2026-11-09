# Reproduction and software checks

Run the model checks from the repository README. Optional numerical checks use the separate manuscript-derived input archive described in analysis/README.txt; the Zenodo sample archive is a different dataset.

| Check | Expected result |
|---|---|
| File integrity | Code manifest passes; --with-model checks the released weight; --with-data checks the optional derived-input manifest and binding |
| Reserved-test summaries and product-cluster intervals | 13,440 values match saved results; rtol 1e-9, atol 1e-10 |
| Climate analyses | 88 cases, 308 coefficient rows and 240 spatial-validation rows; 5,068 numeric comparisons pass at rtol 1e-7, atol 1e-8 |
| Archived object statistics | 6,221 reference and 6,294 predicted objects; areas 107,589.25 and 100,791.25 m² |
| Six exclusion scenarios | Count differences, percentages and cell-area arithmetic agree |
| Inventory validation | 6,157 file records and 17 products; 565 scene records and 558 footprint links |
| Synthetic CPU prediction demo | 256- and 512-pixel uint8 masks with values in {0,255}; demo_report.json has passed=true |
| Core-rule checks | Maximum-cardinality matching, four-neighbour connectivity, area boundaries and input contracts pass |
| Figure 4 | 32 coefficient/interval rows match the supplied source values |

The numerical checks operate on supplied derived inputs. Synthetic examples test interfaces, not tree-detection accuracy. Exclusion arithmetic does not independently validate spatial overlays. Training-source provenance does not establish historical cross-set independence.

The tensor-only export retains all 346 parameter and buffer entries from its bound checkpoint. Its portability checks included matching the two stored binary masks on an existing development anchor. These are implementation checks, not new independent reference data. Current software versions are recorded in config/release_test_environment.json. Historical source compatibility is described in docs/HISTORICAL_CODE_NOTES.md.

Model input/output checks verify a known four-band permutation, agreement of GeoTIFF and NumPy predictions, CRS/grid preservation, known pixel and object confusion counts, and rejection of invalid or misaligned inputs. The sample-index validator checks the 956-pair collection and optional checksums.
