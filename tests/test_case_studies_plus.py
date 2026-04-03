from morphaccel.case_studies import CaseStudyCatalog

def test_case_catalog_builds_service_cases():
    catalog = CaseStudyCatalog()
    cases = catalog.build()
    assert len(cases) >= 6
    assert any((case.name == 'Private RAG Gateway' for case in cases))
    for case in cases:
        assert len(case.graphs) >= 3
        assert len(case.counts) == len(case.graphs)
        assert len(case.gaps_ms) == len(case.graphs)
        assert case.slo_ms > 0

def test_case_manifest_has_expected_columns():
    manifest = CaseStudyCatalog().manifest()
    assert {'case', 'slug', 'slo_ms', 'storyline', 'focus', 'stages', 'prompt'}.issubset(set(manifest.columns))
    assert manifest.shape[0] >= 6
