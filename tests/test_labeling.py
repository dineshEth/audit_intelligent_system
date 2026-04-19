from pathlib import Path



from audit_intelligence.config import Settings

from audit_intelligence.ingestion.bank_statement import parse_bank_statement

from audit_intelligence.labeling.hybrid_labeler import HybridBankStatementLabeler

from audit_intelligence.llm.local_llm import LocalLLM





def test_hybrid_labeler_assigns_categories(tmp_path):

    settings = Settings(project_root=tmp_path, mongodb_uri="mongomock://localhost", mongodb_db_name="test_db")

    settings.ensure_directories()



    sample = Path(__file__).resolve().parents[1] / "datasets" / "raw_samples" / "sample_bank_statement.csv"

    df = parse_bank_statement(sample)



    labeler = HybridBankStatementLabeler(LocalLLM(""), enable_llm_refinement=False)

    labeled_df, rows = labeler.label_dataframe(df)



    assert not labeled_df.empty

    assert "CATEGORY" in labeled_df.columns

    assert len(rows) == len(df)

    assert "FOOD" in set(labeled_df["CATEGORY"]) or "INCOME" in set(labeled_df["CATEGORY"])

