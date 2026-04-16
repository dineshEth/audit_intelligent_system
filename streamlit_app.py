from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from audit_intelligence.config import get_settings
from audit_intelligence.finetune.watcher import DataWatcher
from audit_intelligence.services.dashboard_service import DashboardService
from audit_intelligence.services.pipeline_service import PipelineService


st.set_page_config(page_title="Audit Intelligence System", layout="wide")


@st.cache_resource
def get_service() -> PipelineService:
    return PipelineService(get_settings())


service = get_service()
dashboard_service = DashboardService(service.repositories)
watcher = DataWatcher(service.repositories, service.settings)

st.title("Audit Intelligence System")
st.caption("Offline-first audit assistant with LangGraph orchestration, MongoDB persistence, bank statement labeling, and local PEFT fine-tuning.")

tab_upload, tab_query, tab_finetune, tab_dashboard = st.tabs(
    ["Upload & Analyze", "Query", "Fine-Tuning", "Dashboard"]
)

with tab_upload:
    st.subheader("Upload a document")
    uploaded = st.file_uploader("Choose a file", type=["csv", "txt", "md", "json", "pdf"])
    user_query = st.text_area(
        "Analysis request",
        value="Analyze this document for audit review, label bank transactions if present, and produce a report.",
    )
    if st.button("Run analysis", use_container_width=True):
        if uploaded is None:
            st.warning("Please upload a file first.")
        else:
            try:
                result = service.process_upload(uploaded.name, uploaded.getvalue(), user_query=user_query)
                document = result["document"]
                response = result["response"]
                review = result["review"]

                col1, col2, col3 = st.columns(3)
                col1.metric("Document type", document["doc_type"])
                col2.metric("Review confidence", review.get("confidence", 0.0))
                col3.metric("Q&A pairs", response.get("qa_pair_count", 0))

                st.markdown("### Summary")
                st.write(response.get("summary_text", ""))

                st.markdown("### Metrics")
                st.json(response.get("metrics", {}))

                if response.get("references"):
                    st.markdown("### References")
                    for ref in response["references"]:
                        st.write(f"- score={ref.get('score')} | {ref.get('text')}")

                label_info = response.get("labels", {})
                if label_info.get("export_paths"):
                    st.markdown("### Labeling outputs")
                    st.json(label_info["export_paths"])
                    human_csv = label_info["export_paths"].get("human_csv")
                    if human_csv and Path(human_csv).exists():
                        st.dataframe(pd.read_csv(human_csv), use_container_width=True)

                if response.get("chart_paths"):
                    st.markdown("### Charts")
                    for chart in response["chart_paths"]:
                        st.image(chart, use_container_width=True)

                if response.get("report_path"):
                    st.success(f"Report generated: {response['report_path']}")

                if response.get("fine_tune_run"):
                    st.markdown("### Fine-tuning trigger")
                    st.json(response["fine_tune_run"])

                with st.expander("Trace"):
                    st.json(result.get("trace", []))
            except Exception as exc:
                st.error(f"Processing failed: {exc}")

with tab_query:
    st.subheader("Ask the audit corpus")
    docs = service.latest_documents(limit=100)
    doc_options = {f"{doc.file_name} ({doc.doc_type})": doc.id for doc in docs}
    selected = st.multiselect("Limit search to documents", options=list(doc_options.keys()))
    query = st.text_input("Query", value="What are the key anomalies and totals?")
    if st.button("Run query", key="run_query", use_container_width=True):
        try:
            selected_ids = [doc_options[item] for item in selected] if selected else None
            result = service.answer_query(query, selected_ids)
            st.markdown("### Response")
            st.write(result["response"].get("summary_text", ""))
            st.markdown("### Review")
            st.json(result.get("review", {}))
            st.markdown("### References")
            for ref in result["response"].get("references", []):
                st.write(f"- score={ref.get('score')} | {ref.get('text')}")
        except Exception as exc:
            st.error(f"Query failed: {exc}")

with tab_finetune:
    st.subheader("Local fine-tuning controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Scan for new data and fine-tune", use_container_width=True):
            try:
                outcome = watcher.scan_and_maybe_finetune()
                if outcome is None:
                    st.info("No new labeled or Q&A data detected.")
                else:
                    st.json(outcome.model_dump())
            except Exception as exc:
                st.error(f"Fine-tuning failed: {exc}")

    with col2:
        st.write("Tracked folders:")
        st.code(f"{service.settings.labeled_data_dir}\n{service.settings.qa_data_dir}")

    st.markdown("### Recent model runs")
    model_runs = service.repositories.model_runs.find_many({}, limit=10, sort=("created_at", -1))
    if model_runs:
        st.dataframe(pd.DataFrame([run.model_dump() for run in model_runs]), use_container_width=True)
    else:
        st.info("No model runs logged yet.")

with tab_dashboard:
    st.subheader("System dashboard")
    snapshot = dashboard_service.snapshot()
    counts = snapshot["counts"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", counts["documents"])
    col2.metric("Chunks", counts["chunks"])
    col3.metric("Labels", counts["labels"])
    col4.metric("Q&A pairs", counts["qa_pairs"])

    col5, col6, col7 = st.columns(3)
    col5.metric("Queries", counts["queries"])
    col6.metric("Model runs", counts["model_runs"])
    col7.metric("Logs", counts["logs"])

    st.markdown("### Latest model run")
    st.json(snapshot.get("latest_model_run") or {})

    st.markdown("### Recent logs")
    if snapshot["recent_logs"]:
        st.dataframe(pd.DataFrame(snapshot["recent_logs"]), use_container_width=True)
    else:
        st.info("No logs available yet.")
