from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..analysis.analyzer import FinancialAnalyzer
from ..analysis.charts import ChartBuilder
from ..analysis.reports import ReportBuilder
from ..finetune.watcher import DataWatcher
from ..labeling.exporters import LabelingExporter
from ..labeling.hybrid_labeler import HybridBankStatementLabeler
from ..llm.qa_generation import QAGenerator
from ..rag.graph_relations import build_transaction_graph, graph_summary
from ..schemas import LabelRecord, QAPairRecord
from .base import BaseAgent


class LabelingAgent(BaseAgent):
    name = "labeling"

    def __init__(self, repositories, settings, llm) -> None:
        super().__init__(repositories, settings)
        self.llm = llm
        self.labeler = HybridBankStatementLabeler(llm, enable_llm_refinement=settings.enable_llm_label_refinement)
        self.exporter = LabelingExporter(settings)
        self.analyzer = FinancialAnalyzer()
        self.chart_builder = ChartBuilder(settings)
        self.report_builder = ReportBuilder(settings)
        self.qa_generator = QAGenerator(llm)
        self.data_watcher = DataWatcher(repositories, settings)

    def run(self, documents: List[Any], execution_output: Dict[str, Any], review_output: Dict[str, Any]) -> Dict[str, Any]:
        summary_text = execution_output.get("summary_text", "")
        references = execution_output.get("references", [])
        bank_df = execution_output.get("bank_df")
        combined_text = execution_output.get("combined_text", "")
        chart_paths: List[str] = []
        label_info: Dict[str, Any] = {}
        metrics = execution_output.get("analysis", {}) or {}
        qa_pairs = []

        if bank_df is not None and not bank_df.empty:
            labeled_df, label_rows = self.labeler.label_dataframe(bank_df)
            source_name = documents[0].file_name if documents else "bank_statement"
            export_paths = self.exporter.export(source_name, labeled_df, [item.model_dump() for item in label_rows])

            label_record = LabelRecord(
                document_id=documents[0].id if documents else "",
                source_file=source_name,
                labels=label_rows,
                export_paths=export_paths,
            )
            self.repositories.labels.insert(label_record)

            metrics = self.analyzer.analyze_transactions(labeled_df)
            graph_info = graph_summary(build_transaction_graph(labeled_df))
            metrics["relationship_graph"] = graph_info
            chart_paths = self.chart_builder.build_transaction_charts(labeled_df, Path(source_name).stem)

            qa_pairs = self.qa_generator.build_pairs(combined_text, labeled_df, metrics)
            for qa in qa_pairs:
                self.repositories.qa_pairs.insert(
                    QAPairRecord(
                        document_id=documents[0].id if documents else None,
                        question=qa["question"],
                        answer=qa["answer"],
                        source_type="generated",
                    )
                )

            label_info = {
                "row_count": len(labeled_df),
                "export_paths": export_paths,
            }
        else:
            qa_pairs = self.qa_generator.build_pairs(combined_text)
            for qa in qa_pairs:
                self.repositories.qa_pairs.insert(
                    QAPairRecord(
                        document_id=documents[0].id if documents else None,
                        question=qa["question"],
                        answer=qa["answer"],
                        source_type="generated",
                    )
                )

        report_path = self.report_builder.build_audit_report(
            source_name=documents[0].file_name if documents else "document",
            summary_text=summary_text,
            metrics=metrics,
            references=references,
            chart_paths=chart_paths,
        )

        finetune_result = self.data_watcher.scan_and_maybe_finetune()
        finetune_payload = finetune_result.model_dump() if finetune_result else None

        output = {
            "summary_text": summary_text,
            "metrics": metrics,
            "references": references,
            "review": review_output,
            "labels": label_info,
            "qa_pair_count": len(qa_pairs),
            "chart_paths": chart_paths,
            "report_path": report_path,
            "fine_tune_run": finetune_payload,
        }
        self.log(
            "label_and_export",
            status="success",
            input_payload={"document_ids": [doc.id for doc in documents]},
            output_payload={
                "report_path": report_path,
                "chart_count": len(chart_paths),
                "qa_pair_count": len(qa_pairs),
                "fine_tune_status": finetune_payload.get("status") if finetune_payload else None,
            },
        )
        return output
