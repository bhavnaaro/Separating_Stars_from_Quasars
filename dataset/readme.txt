The following catalogs are present in the root folder.

Catalog 1: North Galactic Region Only

Selected only samples that have fuv. Populated the entire feature list (with
pairwise differences). Ran Random Forests (RF) without upsampling to generate
predicted labels.
-------------------------------------------------------------------------------

Catalog 2: Equatorial Region Only

Selected only samples that have fuv. Populated the entire feature list (with
pairwise differences). Ran RF without upsampling to generate predicted labels.
-------------------------------------------------------------------------------


Catalog 3: North Galactic Region and Equatorial Region Combined

Selected only samples that have fuv. Populated the entire feature list (with
pairwise differences). Ran RF without upsampling to generate predicted labels.
-------------------------------------------------------------------------------


Catalog 4: Removed fuv and fuv-related features

Populate the entire feature list (even with samples that don't have fuv, with
pairwise differences). Run RF without upsampling to generate predicted labels.
-------------------------------------------------------------------------------

As per the nomenclature in the published paper, the three ranges in which we
divide our catalogs are as follows:

1. R1: redshift <= 0.0033
2. R2: 0.0033 < redshift < 0.004
3. R3: redshift >= 0.004

The sub-catalogs present in each catalog folder consist of catalogs of samples
in each of the above ranges, as well as separate catalogs for correctly
classified as well as misclassified samples in each range.
